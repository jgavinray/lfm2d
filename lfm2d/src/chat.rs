//! Multi-turn chats rendered as the generative checkpoint's chat template
//! renders them, with `preserve_thinking` on.
//!
//! The template (pinned by sha256 in `Checkpoint::load`) is reproduced byte for
//! byte for the subset a daemon needs: a system prompt plus tools, then user,
//! assistant (`thinking`, `content`, `tool_calls`) and tool turns, and the
//! generation prompt. `tests/chat_template.rs` holds it to fixtures that
//! transformers' `apply_chat_template` rendered from the same template
//! (`tests/reference/dump_chat_template.py`).
//!
//! With `preserve_thinking` on, no turn's bytes depend on any later turn, so a
//! chat renders as a head ([`render_head`]) followed by one segment per message
//! ([`Message::render`]), and rendering `messages[..k]` is a byte prefix of
//! rendering `messages[..k + 1]`. Turn-boundary state rests on that.
//!
//! **Continuing a chat the model wrote.** An assistant segment ends with
//! [`TURN_END`], `<|im_end|>\n`. The model generates `<|im_end|>` (its EOS) and
//! stops there, so whoever keeps the generated ids and appends the next turn
//! must append [`AFTER_EOS`] first. Re-rendering the turn from its text gives
//! the same bytes, but not necessarily the ids the model generated.
//!
//! Content carries no model control tokens; the renderer supplies them, and
//! refuses (never escapes) text that would forge one.
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, MapAccess, SeqAccess, Visitor},
    ser::{self, SerializeMap},
};
use std::fmt;

/// The checkpoint's `bos_token`, which opens every rendering.
pub const BOS: &str = "<|startoftext|>";
/// `add_generation_prompt`: the opening of an assistant turn.
pub const GENERATION_PROMPT: &str = "<|im_start|>assistant\n";
/// What closes every turn.
pub const TURN_END: &str = "<|im_end|>\n";
/// What follows a generated `<|im_end|>` to close the assistant turn the way
/// the template does. See the module docs.
pub const AFTER_EOS: &str = "\n";
/// transformers' `continue_final_message` sentinel. The template strips it from
/// the end of an assistant's content and writes it back after the tool calls;
/// assistant content carrying it anywhere (with or without the space) is
/// refused rather than reproduced.
pub const CONTINUE_FINAL_MESSAGE_TAG: &str = "CONTINUE_FINAL_MESSAGE_TAG ";

/// Substrings that would tokenize as model control tokens. Every added token in
/// the tokenizer starts with `<|` except the other three, which
/// `tests/chat_template.rs` checks against the real tokenizer.
pub const CONTROL_MARKERS: [&str; 4] = ["<|", "<think>", "</think>", "<image>"];

/// The first of [`CONTROL_MARKERS`] that `text` carries. Every check on text
/// bound for a prompt goes through this, so the list has one home.
pub fn control_marker_in(text: &str) -> Option<&'static str> {
    CONTROL_MARKERS.into_iter().find(|m| text.contains(m))
}

/// A chat: the system prompt, the tools the system turn lists, and the turns
/// after it. An empty system prompt with no tools renders no system turn.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Chat {
    #[serde(default)]
    pub system: String,
    /// Tool schemas, each a JSON object rendered with the template's `tojson`.
    #[serde(default)]
    pub tools: Vec<TemplateValue>,
    pub messages: Vec<Message>,
}

/// One turn after the system turn. The template renders any other role (a
/// second `system`, say) as a plain turn; none is accepted here.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "role", rename_all = "lowercase", deny_unknown_fields)]
pub enum Message {
    User {
        content: String,
    },
    /// `thinking` renders as `<think>{thinking}</think>` with no newlines
    /// added; `Some("")` still writes the pair. `content` absent and empty
    /// render the same. `tool_calls` renders after the content.
    Assistant {
        #[serde(default)]
        thinking: Option<String>,
        #[serde(default)]
        content: Option<String>,
        #[serde(default)]
        tool_calls: Option<Vec<ToolCall>>,
    },
    /// A tool's result. May be empty.
    Tool {
        content: String,
    },
}

/// One call in an assistant turn, rendered `name(key=value, ...)` in the
/// arguments' own order.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(try_from = "WireToolCall")]
pub struct ToolCall {
    pub name: String,
    pub arguments: Vec<(String, TemplateValue)>,
}

/// The OpenAI/transformers shape: `{"type": "function", "function": {"name",
/// "arguments"}}`, with `type` optional. `arguments` is an object, never a
/// JSON-encoded string: the template calls `.items()` on it.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireToolCall {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    function: WireFunction,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireFunction {
    name: String,
    arguments: TemplateValue,
}
impl TryFrom<WireToolCall> for ToolCall {
    type Error = String;
    fn try_from(wire: WireToolCall) -> Result<Self, String> {
        if wire.kind.as_deref().is_some_and(|k| k != "function") {
            return Err("a tool call's type must be \"function\"".into());
        }
        let TemplateValue::Map(arguments) = wire.function.arguments else {
            return Err("a tool call's arguments must be a JSON object".into());
        };
        Ok(ToolCall { name: wire.function.name, arguments })
    }
}

/// A JSON value as the template sees it after Python's `json.loads`: object
/// keys keep document order, and a number stays an int or a float as written.
///
/// Deserialize it from request text, never through `serde_json::Value`, which
/// sorts object keys and would reorder every argument.
///
/// A number is refused where our parse could differ from Python's
/// ([`check_float`]). The visitor sees only the parsed `f64`, never the text,
/// so that rule is stated on the value.
#[derive(Clone, Debug, PartialEq)]
pub enum TemplateValue {
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    List(Vec<TemplateValue>),
    Map(Vec<(String, TemplateValue)>),
}

impl TemplateValue {
    /// A map's value under `key`; `None` for a missing key or a non-map.
    pub fn get(&self, key: &str) -> Option<&TemplateValue> {
        match self {
            TemplateValue::Map(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

/// Writes the value back as JSON, keys in their own order.
impl Serialize for TemplateValue {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            TemplateValue::Null => serializer.serialize_unit(),
            TemplateValue::Bool(b) => serializer.serialize_bool(*b),
            TemplateValue::Int(i) => match (i64::try_from(*i), u64::try_from(*i)) {
                (Ok(i), _) => serializer.serialize_i64(i),
                (_, Ok(u)) => serializer.serialize_u64(u),
                _ => Err(ser::Error::custom("an integer beyond 64 bits")),
            },
            TemplateValue::Float(f) => serializer.serialize_f64(*f),
            TemplateValue::Str(s) => serializer.serialize_str(s),
            TemplateValue::List(items) => items.serialize(serializer),
            TemplateValue::Map(entries) => {
                let mut map = serializer.serialize_map(Some(entries.len()))?;
                for (k, v) in entries {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for TemplateValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = TemplateValue;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON value")
            }
            fn visit_unit<E>(self) -> Result<TemplateValue, E> {
                Ok(TemplateValue::Null)
            }
            fn visit_none<E>(self) -> Result<TemplateValue, E> {
                Ok(TemplateValue::Null)
            }
            fn visit_bool<E>(self, v: bool) -> Result<TemplateValue, E> {
                Ok(TemplateValue::Bool(v))
            }
            fn visit_i64<E>(self, v: i64) -> Result<TemplateValue, E> {
                Ok(TemplateValue::Int(v.into()))
            }
            fn visit_u64<E>(self, v: u64) -> Result<TemplateValue, E> {
                Ok(TemplateValue::Int(v.into()))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> Result<TemplateValue, E> {
                check_float(v).map_err(E::custom)?;
                Ok(TemplateValue::Float(v))
            }
            fn visit_str<E>(self, v: &str) -> Result<TemplateValue, E> {
                Ok(TemplateValue::Str(v.to_owned()))
            }
            fn visit_string<E>(self, v: String) -> Result<TemplateValue, E> {
                Ok(TemplateValue::Str(v))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<TemplateValue, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(TemplateValue::List(items))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<TemplateValue, A::Error> {
                let mut entries: Vec<(String, TemplateValue)> = Vec::new();
                while let Some((key, value)) = map.next_entry::<String, TemplateValue>()? {
                    // Python keeps the first key's position and the last value;
                    // a caller who sent both meant something we cannot know.
                    if entries.iter().any(|(k, _)| *k == key) {
                        return Err(de::Error::custom(format!("duplicate key {key:?}")));
                    }
                    entries.push((key, value));
                }
                Ok(TemplateValue::Map(entries))
            }
        }
        deserializer.deserialize_any(V)
    }
}

/// `<|startoftext|>` and, when there is a system prompt or any tool, the system
/// turn: the prompt, then `List of tools: [...]` on its own line.
pub fn render_head(system: &str, tools: &[TemplateValue]) -> Result<String, String> {
    let mut rendered = Vec::with_capacity(tools.len());
    for tool in tools {
        // transformers refuses a tool that is not a mapping before the template
        // runs, so the template's own string-tool branch is unreachable.
        if !matches!(tool, TemplateValue::Map(_)) {
            return Err("each tool must be a JSON object".into());
        }
        let mut json = String::new();
        tojson(tool, &mut json);
        refuse_control("a tool schema", &json)?;
        rendered.push(json);
    }
    head_with_rendered_tools(system, &rendered)
}

/// [`render_head`] over tools already rendered to text: the template's
/// `ns.system_prompt` assembly.
fn head_with_rendered_tools(system: &str, tools: &[String]) -> Result<String, String> {
    refuse_control("the system prompt", system)?;
    let mut prompt = system.to_owned();
    if !tools.is_empty() {
        if !prompt.is_empty() {
            prompt.push('\n');
        }
        prompt.push_str("List of tools: [");
        prompt.push_str(&tools.join(", "));
        prompt.push(']');
    }
    if prompt.is_empty() {
        return Ok(BOS.to_owned());
    }
    Ok(format!("{BOS}<|im_start|>system\n{prompt}{TURN_END}"))
}

impl Message {
    /// This turn's bytes, exactly as the template appends them.
    pub fn render(&self) -> Result<String, String> {
        match self {
            Message::User { content } => {
                if content.trim().is_empty() {
                    return Err("a user turn must not be empty".into());
                }
                refuse_control("a user turn", content)?;
                Ok(format!("<|im_start|>user\n{content}{TURN_END}"))
            }
            Message::Tool { content } => {
                refuse_control("a tool turn", content)?;
                Ok(format!("<|im_start|>tool\n{content}{TURN_END}"))
            }
            Message::Assistant { thinking, content, tool_calls } => {
                if thinking.is_none() && content.is_none() && tool_calls.is_none() {
                    return Err("an assistant turn must not be empty: give thinking, content or tool_calls".into());
                }
                let mut text = GENERATION_PROMPT.to_owned();
                if let Some(thinking) = thinking {
                    refuse_control("an assistant's thinking", thinking)?;
                    text.push_str("<think>");
                    text.push_str(thinking);
                    text.push_str("</think>");
                }
                if let Some(content) = content {
                    refuse_control("an assistant's content", content)?;
                    if content.contains(CONTINUE_FINAL_MESSAGE_TAG.trim_end()) {
                        return Err(format!(
                            "assistant content must not carry {CONTINUE_FINAL_MESSAGE_TAG:?}: the \
                             template moves it after the tool calls"
                        ));
                    }
                    text.push_str(content);
                }
                if let Some(calls) = tool_calls {
                    if calls.is_empty() {
                        return Err("tool_calls must not be empty; omit it instead".into());
                    }
                    let calls = calls.iter().map(ToolCall::render).collect::<Result<Vec<_>, _>>()?;
                    text.push_str("<|tool_call_start|>[");
                    text.push_str(&calls.join(", "));
                    text.push_str("]<|tool_call_end|>");
                }
                text.push_str(TURN_END);
                Ok(text)
            }
        }
    }
}

impl ToolCall {
    /// `name(key=value, ...)`: a string value in single quotes with nothing
    /// escaped, an object through `tojson`, anything else through Jinja's
    /// `| string`, which is Python's `str()`.
    fn render(&self) -> Result<String, String> {
        if self.name.is_empty() {
            return Err("a tool call's name must not be empty".into());
        }
        refuse_control("a tool call's name", &self.name)?;
        let mut text = format!("{}(", self.name);
        for (i, (key, value)) in self.arguments.iter().enumerate() {
            refuse_control("a tool call's argument name", key)?;
            refuse_control_within("a tool call's argument", value)?;
            if i > 0 {
                text.push_str(", ");
            }
            text.push_str(key);
            text.push('=');
            match value {
                TemplateValue::Str(s) => {
                    text.push('\'');
                    text.push_str(s);
                    text.push('\'');
                }
                TemplateValue::Map(_) => tojson(value, &mut text),
                _ => py_repr(value, &mut text)?,
            }
        }
        text.push(')');
        Ok(text)
    }
}

fn refuse_control(what: &str, text: &str) -> Result<(), String> {
    match control_marker_in(text) {
        Some(marker) => Err(format!(
            "{what} carries {marker:?}, a model control token; the renderer writes those"
        )),
        None => Ok(()),
    }
}

/// Every string and key inside a value, before any rendering escapes it.
fn refuse_control_within(what: &str, value: &TemplateValue) -> Result<(), String> {
    match value {
        TemplateValue::Str(s) => refuse_control(what, s),
        TemplateValue::List(items) => items.iter().try_for_each(|v| refuse_control_within(what, v)),
        TemplateValue::Map(entries) => entries.iter().try_for_each(|(k, v)| {
            refuse_control(what, k)?;
            refuse_control_within(what, v)
        }),
        _ => Ok(()),
    }
}

/// transformers' `tojson`: `json.dumps(x, ensure_ascii=False)` with the
/// default `", "` and `": "` separators and keys in insertion order. Not
/// jinja2's own filter, which escapes `<`, `>`, `&` and `'` for HTML.
fn tojson(value: &TemplateValue, out: &mut String) {
    match value {
        TemplateValue::Null => out.push_str("null"),
        TemplateValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        TemplateValue::Int(i) => out.push_str(&i.to_string()),
        TemplateValue::Float(f) => out.push_str(&py_float_repr(*f)),
        TemplateValue::Str(s) => json_string(s, out),
        TemplateValue::List(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                tojson(item, out);
            }
            out.push(']');
        }
        TemplateValue::Map(entries) => {
            out.push('{');
            for (i, (key, item)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                json_string(key, out);
                out.push_str(": ");
                tojson(item, out);
            }
            out.push('}');
        }
    }
}

/// Python's `json` string encoding with `ensure_ascii=False`: quote, backslash
/// and the ASCII controls are escaped, everything else is written as is.
fn json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr()`, which is what `str()` gives for everything but a string.
/// Inside a list, strings and dict keys are repr'd too.
fn py_repr(value: &TemplateValue, out: &mut String) -> Result<(), String> {
    match value {
        TemplateValue::Null => out.push_str("None"),
        TemplateValue::Bool(b) => out.push_str(if *b { "True" } else { "False" }),
        TemplateValue::Int(i) => out.push_str(&i.to_string()),
        TemplateValue::Float(f) => out.push_str(&py_float_repr(*f)),
        TemplateValue::Str(s) => py_str_repr(s, out)?,
        TemplateValue::List(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_repr(item, out)?;
            }
            out.push(']');
        }
        TemplateValue::Map(entries) => {
            out.push('{');
            for (i, (key, item)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_str_repr(key, out)?;
                out.push_str(": ");
                py_repr(item, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// Python's string `repr()`: single quotes unless the string holds a single
/// quote and no double quote, and escapes for the quote in use, backslash and
/// non-printable characters.
///
/// Which non-ASCII characters are printable comes from the Unicode database of
/// the Python that ran the template, so it moves between Python versions. ASCII
/// and the C1 controls are exact; beyond them only letters and digits are
/// accepted (printable in every Python that has them assigned; one newer than
/// that Python's Unicode would be escaped there and written here), and anything
/// else (a no-break space, an emoji, a format character) is refused rather than
/// guessed at.
fn py_str_repr(s: &str, out: &mut String) -> Result<(), String> {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            ' '..='~' => out.push(c),
            // The C0 and C1 controls: the same escape in every Python.
            c if c.is_ascii() || c.is_control() => out.push_str(&format!("\\x{:02x}", c as u32)),
            c if c.is_alphanumeric() => out.push(c),
            c => {
                return Err(format!(
                    "a string inside a list argument carries {c:?} (U+{:04X}); the template writes \
                     it through Python's repr, whose escaping of non-ASCII beyond letters and \
                     digits depends on the Python version",
                    c as u32
                ));
            }
        }
    }
    out.push(quote);
    Ok(())
}

/// The shortest round-tripping decimal digits of `|f|` and the exponent of the
/// first: `|f| = d.ddd × 10^exponent`.
fn shortest_digits(f: f64) -> (String, i32) {
    let sci = format!("{:e}", f.abs());
    let (mantissa, exponent) = sci.split_once('e').expect("`{:e}` always writes an exponent");
    let digits = mantissa.chars().filter(|c| *c != '.').collect();
    (digits, exponent.parse().expect("`{:e}` writes an integer exponent"))
}

/// Floats whose parse could differ from Python's `json.loads`, refused: a
/// whole number at or beyond 2^64 (or at or below -2^63). An integer written
/// there overflows serde_json's `u64`/`i64` and arrives as a float, where
/// Python keeps an int (`100000000000000000000`, not `1e+20`), and from the
/// value alone the two cannot be told apart.
///
/// Every other float is read exactly: the crate turns on serde_json's
/// `float_roundtrip`, whose parse is correctly rounded as Python's is. Without
/// it serde_json computes `D as f64` and one multiply or divide by `10^e`, off
/// by an ulp past `D > 2^53` or `|e| > 22` (it read `6.02e-23` as
/// `6.019999999999999e-23`); `tests/chat_template.rs` holds the parse to
/// Rust's own, which is exact, over a sample of doubles.
fn check_float(f: f64) -> Result<(), String> {
    if !f.is_finite() {
        return Err("a float must be finite".into());
    }
    if f.fract() == 0.0 && (f >= 18446744073709551616.0 || f <= -9223372036854775808.0) {
        return Err(format!(
            "the float {} is a whole number beyond 64-bit integers, where an integer arrives as a \
             float too, so which one Python would render is unknown",
            py_float_repr(f)
        ));
    }
    Ok(())
}

/// Python's `float.__repr__`: the shortest digits that round-trip (as Rust
/// finds them too), positional when the decimal point falls within
/// `-4 < decpt <= 16`, otherwise `d.ddde±XX` with at least two exponent digits;
/// a whole number keeps `.0`.
fn py_float_repr(f: f64) -> String {
    let (digits, exponent) = shortest_digits(f);
    let decpt = exponent + 1;
    let n = digits.len() as i32;
    let mut out = String::new();
    if f.is_sign_negative() {
        out.push('-');
    }
    if -4 < decpt && decpt <= 16 {
        if decpt <= 0 {
            out.push_str("0.");
            out.push_str(&"0".repeat((-decpt) as usize));
            out.push_str(&digits);
        } else if decpt >= n {
            out.push_str(&digits);
            out.push_str(&"0".repeat((decpt - n) as usize));
            out.push_str(".0");
        } else {
            out.push_str(&digits[..decpt as usize]);
            out.push('.');
            out.push_str(&digits[decpt as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        if n > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let sign = if exponent < 0 { '-' } else { '+' };
        out.push_str(&format!("e{sign}{:02}", exponent.abs()));
    }
    out
}

impl Chat {
    pub fn render_head(&self) -> Result<String, String> {
        render_head(&self.system, &self.tools)
    }

    /// The head, then one segment per message. Their concatenation is
    /// [`Chat::render`] without the generation prompt.
    pub fn render_segments(&self) -> Result<Vec<String>, String> {
        let mut segments = vec![self.render_head()?];
        for message in &self.messages {
            segments.push(message.render()?);
        }
        Ok(segments)
    }

    /// The whole chat; `add_generation_prompt` appends [`GENERATION_PROMPT`].
    pub fn render(&self, add_generation_prompt: bool) -> Result<String, String> {
        let mut text = self.render_segments()?.concat();
        if add_generation_prompt {
            text.push_str(GENERATION_PROMPT);
        }
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adjudicator::{PromptSpec, Reasoning};

    /// What `Reasoning::opening` writes after the generation prompt.
    fn opening(reasoning: Reasoning) -> &'static str {
        match reasoning {
            Reasoning::Closed => "<think>\n\n</think>\n",
            Reasoning::Open => "",
        }
    }

    const INPUT: &str = "Email:\nWhere is my order? It's late — ¿dónde está?";

    #[test]
    fn a_schema_spec_single_turn_prompt_is_a_system_and_user_chat() {
        // The system prompt here is the spec's instructions plus its stated
        // schema, which `PromptSpec` assembles; the chat renderer owns the
        // framing around it.
        let p: PromptSpec =
            serde_json::from_str(include_str!("../tests/fixtures/specs/email-triage-v1.json")).unwrap();
        assert!(p.output_schema.is_some() && p.tools.is_empty());
        assert_eq!(p.reasoning, Reasoning::Closed);
        let prefix = p.render_prefix().unwrap();
        let system = prefix
            .strip_prefix("<|startoftext|><|im_start|>system\n")
            .and_then(|s| s.strip_suffix("<|im_end|>\n"))
            .unwrap();
        let chat = Chat {
            system: system.into(),
            tools: vec![],
            messages: vec![Message::User { content: INPUT.into() }],
        };
        assert_eq!(
            chat.render(true).unwrap() + opening(p.reasoning),
            prefix + &p.render_user_turn(INPUT)
        );
    }

    #[test]
    fn a_tools_spec_single_turn_prompt_is_a_system_and_user_chat() {
        let mut p: PromptSpec =
            serde_json::from_str(include_str!("../tests/fixtures/specs/email-triage-tools-v1.json")).unwrap();
        assert!(p.output_schema.is_none() && !p.tools.is_empty());
        for reasoning in [Reasoning::Open, Reasoning::Closed] {
            p.reasoning = reasoning;
            let chat = Chat {
                system: p.system.clone(),
                tools: p.tools.clone(),
                messages: vec![Message::User { content: INPUT.into() }],
            };
            assert_eq!(
                chat.render(true).unwrap() + opening(reasoning),
                p.render_prefix().unwrap() + &p.render_user_turn(INPUT),
                "{reasoning:?}"
            );
        }
    }

    #[test]
    fn floats_render_as_python_repr() {
        for (f, want) in [
            (0.1, "0.1"),
            (1e22, "1e+22"),
            (5e-324, "5e-324"),
            (f64::MAX, "1.7976931348623157e+308"),
            (123456789012345680.0, "1.2345678901234568e+17"),
            (1e-4, "0.0001"),
            (0.000123, "0.000123"),
            (12345.678, "12345.678"),
            (100.0, "100.0"),
            (-1.5, "-1.5"),
            (9999999999999998.0, "9999999999999998.0"),
            (-1e-300, "-1e-300"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
        ] {
            assert_eq!(py_float_repr(f), want, "{f:e}");
        }
    }

    #[test]
    fn strings_inside_lists_render_as_python_repr() {
        for (s, want) in [
            ("a'b", r#""a'b""#),
            ("a\"b", r#"'a"b'"#),
            ("a'\"b", r#"'a\'"b'"#),
            ("\x00\x1f\x7f\u{85}", r"'\x00\x1f\x7f\x85'"),
            ("tab\tnl\ncr\rbs\\", r"'tab\tnl\ncr\rbs\\'"),
            ("猫 café ٣", "'猫 café ٣'"),
        ] {
            let mut out = String::new();
            py_str_repr(s, &mut out).unwrap();
            assert_eq!(out, want, "{s:?}");
        }
    }
}
