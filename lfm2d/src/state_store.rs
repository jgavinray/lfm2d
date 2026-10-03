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
//! An entry may be pinned (a held context, `POST /v1/contexts` with `pin`):
//! eviction passes over it, so pins may hold at most half the budget, and an
//! insert that could only fit by evicting a pin is refused before anything
//! is evicted.
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
    pinned: bool,
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
    /// Bytes held by pinned entries.
    pub(crate) fn pinned_bytes(&self) -> usize {
        self.entries.iter().filter(|e| e.pinned).map(|e| e.bytes).sum()
    }
    /// What `id` is charged; `None` when it is not held.
    pub(crate) fn bytes_of(&self, id: &str) -> Option<usize> {
        self.entries.iter().find(|e| e.id == id).map(|e| e.bytes)
    }
    /// Whether `id` is held pinned; `None` when it is not held.
    pub(crate) fn is_pinned(&self, id: &str) -> Option<bool> {
        self.entries.iter().find(|e| e.id == id).map(|e| e.pinned)
    }
    /// Pin or unpin a held entry. Pins may hold at most half the budget: a
    /// pin past that is refused and changes nothing. `Ok(false)` when `id`
    /// is not held.
    pub(crate) fn set_pinned(&mut self, id: &str, pinned: bool) -> Result<bool, String> {
        let pinned_bytes = self.pinned_bytes();
        let budget = self.budget;
        let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) else {
            return Ok(false);
        };
        if pinned && !entry.pinned && pinned_bytes + entry.bytes > budget / 2 {
            return Err(format!(
                "pinning {} bytes would put {} bytes under pins, past half the budget ({} bytes)",
                entry.bytes,
                pinned_bytes + entry.bytes,
                budget / 2
            ));
        }
        entry.pinned = pinned;
        Ok(true)
    }
    /// Drop `id`, pinned or not; returns its value when it was held.
    pub(crate) fn remove(&mut self, id: &str) -> Option<Arc<T>> {
        let at = self.entries.iter().position(|e| e.id == id)?;
        let old = self.entries.remove(at).expect("position came from this deque");
        self.used -= old.bytes;
        Some(old.value)
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
    /// fit, and so is one that could only fit by evicting a pin. Pinned
    /// entries are never evicted, not even by their group's cap.
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
        let pinned = self.pinned_bytes();
        if pinned + bytes > self.budget {
            return Err(format!(
                "an entry of {bytes} bytes does not fit beside {pinned} pinned bytes in a budget of {} bytes",
                self.budget
            ));
        }
        let mut evicted = Vec::new();
        if let Some((group, cap)) = group {
            if cap == 0 {
                return Ok(evicted);
            }
            while self.entries.iter().filter(|e| e.group.as_deref() == Some(group)).count() >= cap {
                let Some(at) = self.entries.iter().position(|e| e.group.as_deref() == Some(group) && !e.pinned)
                else {
                    break;
                };
                let old = self.entries.remove(at).expect("position came from this deque");
                self.used -= old.bytes;
                evicted.push(old.id);
            }
        }
        while self.used + bytes > self.budget {
            let at = self
                .entries
                .iter()
                .position(|e| !e.pinned)
                .expect("pinned + bytes <= budget, so an unpinned entry is left to evict");
            let old = self.entries.remove(at).expect("position came from this deque");
            self.used -= old.bytes;
            evicted.push(old.id);
        }
        self.used += bytes;
        self.entries.push_back(Entry { id, group: group.map(|(g, _)| g.to_owned()), bytes, pinned: false, value });
        Ok(evicted)
    }
    /// Drop every entry whose id fails `keep`, pinned or not; returns how
    /// many went.
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

    #[test]
    fn a_pinned_entry_is_never_evicted_and_pins_hold_at_most_half() {
        let mut store = StateStore::new(100);
        store.insert("p".into(), 40, Arc::new(0)).unwrap();
        assert_eq!(store.set_pinned("p", true), Ok(true));
        assert_eq!(store.is_pinned("p"), Some(true));
        store.insert("a".into(), 40, Arc::new(1)).unwrap();
        // `p` is the least recently used, but pinned: `a` goes instead.
        assert_eq!(store.insert("b".into(), 40, Arc::new(2)).unwrap(), ["a"]);
        assert!(store.peek("p").is_some());
        // Past half the budget under pins: refused, nothing changes.
        let err = store.set_pinned("b", true).unwrap_err();
        assert!(err.contains("half the budget"), "{err}");
        assert_eq!(store.is_pinned("b"), Some(false));
        // Something that could only fit by evicting the pin is refused
        // before anything is evicted.
        let err = store.insert("huge".into(), 61, Arc::new(3)).unwrap_err();
        assert!(err.contains("pinned"), "{err}");
        assert!(store.peek("b").is_some(), "a refusal evicts nothing");
        // Unpinned, it is ordinary again.
        assert_eq!(store.set_pinned("p", false), Ok(true));
        assert_eq!(store.insert("huge".into(), 61, Arc::new(3)).unwrap(), ["p", "b"]);
        assert_eq!(store.set_pinned("gone", true), Ok(false));
    }

    #[test]
    fn remove_drops_an_entry_pinned_or_not_and_frees_its_bytes() {
        let mut store = StateStore::new(100);
        store.insert("a".into(), 30, Arc::new(1)).unwrap();
        store.insert("b".into(), 20, Arc::new(2)).unwrap();
        store.set_pinned("a", true).unwrap();
        assert_eq!(store.remove("a").as_deref(), Some(&1));
        assert_eq!((store.used(), store.pinned_bytes(), store.len()), (20, 0, 1));
        assert!(store.remove("a").is_none());
        assert_eq!(store.is_pinned("a"), None);
    }

    #[test]
    fn a_group_cap_never_evicts_a_pin() {
        let mut store = StateStore::new(100);
        store.insert_in("g1".into(), Some(("g", 1)), 10, Arc::new(1)).unwrap();
        store.set_pinned("g1", true).unwrap();
        // The group is at its cap, but its only member is pinned: it stays,
        // and the group runs over its count rather than lose a pin.
        assert!(store.insert_in("g2".into(), Some(("g", 1)), 10, Arc::new(2)).unwrap().is_empty());
        assert!(store.peek("g1").is_some() && store.peek("g2").is_some());
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
