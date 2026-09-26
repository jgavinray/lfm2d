//! Everything the adjudicator keeps that holds a model state, under a byte
//! budget: least recently used evicted first, never an entry larger than the
//! whole budget. Two stores use it: the chat checkpoints
//! (`crate::chat_session`) and the caches (ready prompts, described states,
//! tail prefixes; `Adjudicator`'s `states`).
//!
//! An entry may also belong to a group with its own count cap: a spec's
//! described states (16) and its ready prompt (1) keep the per-spec counts
//! the menu and `GET /v1/adjudicator` advertise, and the byte budget bounds
//! all of them together. A long state forks to a power-of-two KV capacity,
//! so a count alone never bounded the memory: 16 described states across 8
//! specs at a 32k context could be ~100 GiB.
//!
//! Pure bookkeeping (no model), so the rules are unit-tested here. Values
//! are `Arc`s: a job holds what it got, so an eviction afterwards never
//! pulls a state out from under it.
use std::collections::VecDeque;
use std::sync::Arc;

pub(crate) struct StateStore<T> {
    budget: usize,
    used: usize,
    /// Front: least recently used.
    entries: VecDeque<Entry<T>>,
}
struct Entry<T> {
    id: String,
    group: Option<String>,
    bytes: usize,
    value: Arc<T>,
}
impl<T> StateStore<T> {
    pub(crate) fn new(budget: usize) -> Self {
        Self { budget, used: 0, entries: VecDeque::new() }
    }
    pub(crate) fn budget(&self) -> usize {
        self.budget
    }
    pub(crate) fn used(&self) -> usize {
        self.used
    }
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
    /// The entry, recency untouched.
    pub(crate) fn peek(&self, id: &str) -> Option<&Arc<T>> {
        self.entries.iter().find(|e| e.id == id).map(|e| &e.value)
    }
    /// Every entry, least recently used first, recency untouched.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&str, &Arc<T>)> {
        self.entries.iter().map(|e| (e.id.as_str(), &e.value))
    }
    /// The entry, touched to most recently used.
    pub(crate) fn get(&mut self, id: &str) -> Option<Arc<T>> {
        let at = self.entries.iter().position(|e| e.id == id)?;
        let entry = self.entries.remove(at).expect("position came from this deque");
        let value = entry.value.clone();
        self.entries.push_back(entry);
        Some(value)
    }
    /// Hold `value` under `id`, evicting the least recently used until it
    /// fits; returns the evicted ids. See [`StateStore::insert_in`].
    pub(crate) fn insert(&mut self, id: String, bytes: usize, value: Arc<T>) -> Result<Vec<String>, String> {
        self.insert_in(id, None, bytes, value)
    }
    /// Hold `value` under `id`, first evicting the group's least recently
    /// used while the group is at its cap, then the store's least recently
    /// used until `bytes` fit. An id already held keeps its value (touched):
    /// every id here names one computation, so a second one is the same
    /// state and there is nothing to replace. A group capped at 0 holds
    /// nothing. An entry larger than the whole budget is refused, never
    /// admitted by emptying the store for something that still would not
    /// fit.
    pub(crate) fn insert_in(
        &mut self,
        id: String,
        group: Option<(&str, usize)>,
        bytes: usize,
        value: Arc<T>,
    ) -> Result<Vec<String>, String> {
        if self.get(&id).is_some() {
            return Ok(Vec::new());
        }
        if bytes > self.budget {
            return Err(format!(
                "an entry of {bytes} bytes exceeds the whole budget of {} bytes",
                self.budget
            ));
        }
        let mut evicted = Vec::new();
        if let Some((group, cap)) = group {
            if cap == 0 {
                return Ok(evicted);
            }
            while self.entries.iter().filter(|e| e.group.as_deref() == Some(group)).count() >= cap {
                let at = self
                    .entries
                    .iter()
                    .position(|e| e.group.as_deref() == Some(group))
                    .expect("the group has entries");
                let old = self.entries.remove(at).expect("position came from this deque");
                self.used -= old.bytes;
                evicted.push(old.id);
            }
        }
        while self.used + bytes > self.budget {
            let old = self.entries.pop_front().expect("used > 0 means an entry is held");
            self.used -= old.bytes;
            evicted.push(old.id);
        }
        self.used += bytes;
        self.entries.push_back(Entry { id, group: group.map(|(g, _)| g.to_owned()), bytes, value });
        Ok(evicted)
    }
    /// Drop every entry whose id fails `keep`; returns how many went.
    pub(crate) fn retain(&mut self, keep: impl Fn(&str) -> bool) -> usize {
        let before = self.entries.len();
        let mut used = 0;
        self.entries.retain(|e| {
            let kept = keep(&e.id);
            if kept {
                used += e.bytes;
            }
            kept
        });
        self.used = used;
        before - self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_store_evicts_least_recently_used_by_bytes_not_by_count() {
        let mut store = StateStore::new(100);
        assert!(store.insert("a".into(), 40, Arc::new(1)).unwrap().is_empty());
        assert!(store.insert("b".into(), 40, Arc::new(2)).unwrap().is_empty());
        assert_eq!(store.used(), 80);
        // Touch `a`: `b` is now the least recently used.
        assert_eq!(store.get("a").as_deref(), Some(&1));
        assert_eq!(store.insert("c".into(), 30, Arc::new(3)).unwrap(), ["b"]);
        assert_eq!(store.used(), 70);
        // Many small entries fit where one large one did.
        for i in 0..3 {
            assert!(store.insert(format!("s{i}"), 10, Arc::new(10 + i)).unwrap().is_empty());
        }
        assert_eq!(store.len(), 5);
        assert_eq!(store.used(), 100);
        // A large entry evicts oldest first until it fits, and no further.
        assert_eq!(store.insert("big".into(), 60, Arc::new(9)).unwrap(), ["a", "c"]);
        assert_eq!(store.used(), 90);
        assert!(store.get("s0").is_some());
        assert!(store.get("b").is_none() && store.get("a").is_none());
    }

    #[test]
    fn the_store_keeps_the_first_value_for_an_id_and_refuses_what_cannot_fit() {
        let mut store = StateStore::new(50);
        store.insert("a".into(), 20, Arc::new("first")).unwrap();
        assert!(store.insert("a".into(), 20, Arc::new("second")).unwrap().is_empty());
        assert_eq!(store.used(), 20, "charged once");
        assert_eq!(store.get("a").as_deref(), Some(&"first"));
        let err = store.insert("huge".into(), 51, Arc::new("x")).unwrap_err();
        assert!(err.contains("budget"), "{err}");
        assert_eq!(store.get("a").as_deref(), Some(&"first"), "a refusal evicts nothing");
    }

    #[test]
    fn an_evicted_entry_lives_on_in_the_hands_of_a_job_that_holds_it() {
        let mut store = StateStore::new(10);
        store.insert("a".into(), 10, Arc::new(vec![1, 2, 3])).unwrap();
        let held = store.get("a").unwrap();
        assert_eq!(store.insert("b".into(), 10, Arc::new(vec![])).unwrap(), ["a"]);
        assert_eq!(*held, vec![1, 2, 3]);
    }

    /// A group's cap evicts within the group first, least recently used
    /// first, and the byte budget still bounds everything together.
    #[test]
    fn a_group_cap_evicts_within_its_group_and_bytes_bound_all_groups() {
        let mut store = StateStore::new(100);
        let g = |cap| Some(("spec-a", cap));
        store.insert_in("a1".into(), g(2), 10, Arc::new(1)).unwrap();
        store.insert_in("a2".into(), g(2), 10, Arc::new(2)).unwrap();
        store.insert("other".into(), 10, Arc::new(0)).unwrap();
        store.get("a1");
        // At its cap of 2: `a2` (the group's least recently used) goes, not
        // `other`, which is older but in no group.
        assert_eq!(store.insert_in("a3".into(), g(2), 10, Arc::new(3)).unwrap(), ["a2"]);
        assert!(store.get("other").is_some());
        assert_eq!(store.used(), 30);
        // Bytes still bind across groups, least recently used first: `a1`,
        // then enough room.
        assert_eq!(
            store.insert_in("b1".into(), Some(("spec-b", 16)), 80, Arc::new(4)).unwrap(),
            ["a1"]
        );
        assert_eq!(store.used(), 100);
        // A group capped at 0 holds nothing and evicts nothing.
        assert!(store.insert_in("z".into(), Some(("none", 0)), 1, Arc::new(5)).unwrap().is_empty());
        assert!(store.get("z").is_none());
        assert_eq!(store.retain(|id| id != "other"), 1);
        assert_eq!((store.len(), store.used()), (2, 90));
    }
}
