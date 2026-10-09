"""
Thompson's Construction Engine  -  TAE-II, Theory of Computation (N-PCCCM503T)

Compiler front-end for a basic regular expression (supports  .  |  *  and ( ) ):
    lexer  ->  recursive-descent parser  ->  Thompson construction  ->  epsilon-NFA
plus an epsilon-closure simulator to test valid / invalid input strings.

Run:  streamlit run app.py
"""
from dataclasses import dataclass, field
import pandas as pd
import streamlit as st

EPS = "ε"


# ----------------------------------------------------------------------------
# 1. Front-end: lexer + recursive-descent parser that builds Thompson fragments
# ----------------------------------------------------------------------------
class RegexError(Exception):
    def __init__(self, msg, pos=0):
        super().__init__(msg)
        self.msg, self.pos = msg, pos


@dataclass
class NFA:
    n_states: int = 0
    edges: list = field(default_factory=list)   # (from, symbol or None, to); None = epsilon
    start: int = 0
    accept: int = 0
    postfix: str = ""
    tokens: str = ""
    alphabet: list = field(default_factory=list)
    log: list = field(default_factory=list)      # construction steps


class Builder:
    GRAMMAR_SYMBOLS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.|*()")

    def __init__(self, src):
        self.nfa = NFA()
        self.post = []
        self.toks = []
        for i, ch in enumerate(src):
            if ch == " ":
                continue
            if ch not in self.GRAMMAR_SYMBOLS:
                raise RegexError(f"Invalid character '{ch}'", i)
            self.toks.append((ch, i))
        if not self.toks:
            raise RegexError("Empty expression", 0)
        self.p = 0
        self.src_len = len(src)

    # --- helpers --------------------------------------------------------
    def new_state(self):
        self.nfa.n_states += 1
        return self.nfa.n_states - 1

    def edge(self, f, s, t):
        self.nfa.edges.append((f, s, t))

    def peek(self):
        return self.toks[self.p][0] if self.p < len(self.toks) else None

    def here(self):
        return self.toks[self.p][1] if self.p < len(self.toks) else self.src_len

    def starts_term(self):
        c = self.peek()
        return c is not None and (c.isalnum() or c == "(")

    # --- grammar:  expr -> term ('|' term)* ; term -> factor ('.'? factor)* ;
    #               factor -> atom '*'* ; atom -> symbol | '(' expr ')'
    def expr(self):
        a = self.term()
        while self.peek() == "|":
            self.p += 1
            if not self.starts_term():
                raise RegexError("Missing operand after '|'", self.here())
            b = self.term()
            self.post.append("|")
            s, e = self.new_state(), self.new_state()
            for x, y in ((s, a[0]), (s, b[0]), (a[1], e), (b[1], e)):
                self.edge(x, None, y)
            self.nfa.log.append(("|", "union", f"new q{s}, q{e}; ε: q{s}→q{a[0]}, q{s}→q{b[0]}, q{a[1]}→q{e}, q{b[1]}→q{e}"))
            a = (s, e)
        return a

    def term(self):
        a = self.factor()
        while True:
            if self.peek() == ".":
                self.p += 1
                if not self.starts_term():
                    raise RegexError("Missing operand after '.'", self.here())
            elif not self.starts_term():
                break
            b = self.factor()
            self.post.append(".")
            self.edge(a[1], None, b[0])
            self.nfa.log.append((".", "concatenation", f"ε: q{a[1]}→q{b[0]}"))
            a = (a[0], b[1])
        return a

    def factor(self):
        a = self.atom()
        while self.peek() == "*":
            self.p += 1
            self.post.append("*")
            s, e = self.new_state(), self.new_state()
            for x, y in ((s, a[0]), (s, e), (a[1], a[0]), (a[1], e)):
                self.edge(x, None, y)
            self.nfa.log.append(("*", "Kleene star", f"new q{s}, q{e}; ε: q{s}→q{a[0]}, q{s}→q{e}, q{a[1]}→q{a[0]}, q{a[1]}→q{e}"))
            a = (s, e)
        return a

    def atom(self):
        c = self.peek()
        if c is not None and c.isalnum():
            self.p += 1
            self.post.append(c)
            s, e = self.new_state(), self.new_state()
            self.edge(s, c, e)
            self.nfa.log.append((c, "symbol", f"q{s} --{c}--> q{e}"))
            return (s, e)
        if c == "(":
            opened = self.here()
            self.p += 1
            if self.peek() == ")":
                raise RegexError("Empty parentheses", self.here())
            if not self.starts_term():
                raise RegexError(f"Unexpected '{self.peek() or 'end'}' after '('", self.here())
            a = self.expr()
            if self.peek() != ")":
                raise RegexError(f"Missing ')' for '(' at position {opened}", opened)
            self.p += 1
            return a
        if c is None:
            raise RegexError("Unexpected end of expression", self.src_len)
        if c == ")":
            raise RegexError("Unmatched ')'", self.here())
        raise RegexError(f"Operator '{c}' has no left operand", self.here())

    def build(self):
        s, e = self.expr()
        if self.p < len(self.toks):
            ch = self.peek()
            raise RegexError("Unmatched ')'" if ch == ")" else f"Unexpected '{ch}'", self.here())
        n = self.nfa
        n.start, n.accept = s, e
        n.postfix = " ".join(self.post)
        n.tokens = " ".join(t for t, _ in self.toks)
        n.alphabet = sorted({x for _, x, _ in n.edges if x})
        return n


def compile_regex(src: str) -> NFA:
    return Builder(src).build()


# ----------------------------------------------------------------------------
# 2. Simulation (epsilon-closure based)
# ----------------------------------------------------------------------------
def eps_closure(nfa: NFA, states):
    stack, seen = list(states), set(states)
    while stack:
        u = stack.pop()
        for f, s, t in nfa.edges:
            if f == u and s is None and t not in seen:
                seen.add(t)
                stack.append(t)
    return seen


def simulate(nfa: NFA, w: str):
    cur = eps_closure(nfa, {nfa.start})
    trace = [("start", cur)]
    for ch in w:
        nxt = {t for f, s, t in nfa.edges if f in cur and s == ch}
        cur = eps_closure(nfa, nxt)
        trace.append((ch, cur))
    return nfa.accept in cur, trace


def fmt(states):
    return "{" + ", ".join(f"q{x}" for x in sorted(states)) + "}" if states else "∅"


# ----------------------------------------------------------------------------
# 3. Presentation helpers
# ----------------------------------------------------------------------------
def to_dot(nfa: NFA) -> str:
    lines = ["digraph NFA {", "rankdir=LR;", 'node [shape=circle, fontsize=12];',
             'start_ [shape=point];', f"start_ -> q{nfa.start};",
             f"q{nfa.accept} [shape=doublecircle];"]
    for f, s, t in nfa.edges:
        if s is None:
            lines.append(f'q{f} -> q{t} [label="{EPS}", style=dashed, color=gray40];')
        else:
            lines.append(f'q{f} -> q{t} [label="{s}", fontcolor=blue, penwidth=1.6];')
    lines.append("}")
    return "\n".join(lines)


def transition_table(nfa: NFA) -> pd.DataFrame:
    cols = nfa.alphabet + [EPS]
    rows = []
    for i in range(nfa.n_states):
        label = ("→ " if i == nfa.start else "") + ("* " if i == nfa.accept else "") + f"q{i}"
        row = {"State": label}
        for c in cols:
            key = None if c == EPS else c
            tgt = sorted(t for f, s, t in nfa.edges if f == i and s == key)
            row[c] = "{" + ", ".join(f"q{t}" for t in tgt) + "}" if tgt else "∅"
        rows.append(row)
    return pd.DataFrame(rows)


SAMPLES = {
    "(a|b)*.a.b.b": ["abb", "aabb", "babb", "ab", "abab", "abba", ""],
    "a.b|c*": ["ab", "c", "ccc", "", "abc", "a"],
    "a*.b": ["b", "aaab", "aa", "ba", ""],
    "(a.b)*": ["", "ab", "abab", "aba", "ba"],
}
SUITE = ["a", "a.b", "a|b", "a*", "(a|b)*.a.b.b", "a.b|c*", "(a.(b|c))*", "ab*",
         "(a|b", "a|", "|a", "*a", "a.*b", "a||b", "ab)", "()", "a+b", "", "a."]


# ----------------------------------------------------------------------------
# 4. Streamlit UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Thompson's Construction Engine", page_icon="🔧", layout="wide")
    st.title("🔧 Thompson's Construction Engine")
    st.caption("Regex → compiler front-end → ε-NFA  |  operators: `.` concatenation, `|` union, "
               "`*` Kleene star, `( )` grouping  |  symbols: letters / digits "
               "(adjacent symbols are also concatenated)")

    if "regex" not in st.session_state:
        st.session_state.regex = "(a|b)*.a.b.b"

    st.subheader("1. Enter a regular expression")
    cols = st.columns(len(SAMPLES))
    for c, r in zip(cols, SAMPLES):
        if c.button(r, use_container_width=True):
            st.session_state.regex = r
    regex = st.text_input("Regular expression", key="regex")

    try:
        nfa = compile_regex(regex)
    except RegexError as e:
        st.error(f"✗ Invalid regex: {e.msg} (position {e.pos})")
        st.code(regex + "\n" + " " * e.pos + "^")
        nfa = None

    if nfa:
        st.success("✓ Valid regex — ε-NFA constructed.")
        st.subheader("2. Front-end output")
        st.markdown(f"**Tokens:** `{nfa.tokens}`  \n**Postfix:** `{nfa.postfix}`")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("States", nfa.n_states)
        m2.metric("Transitions", len(nfa.edges))
        m3.metric("Start / Accept", f"q{nfa.start} / q{nfa.accept}")
        m4.metric("Alphabet", "{" + ",".join(nfa.alphabet) + "}")

        with st.expander("Construction steps (one per postfix operator)", expanded=False):
            st.dataframe(pd.DataFrame(nfa.log, columns=["Postfix symbol", "Rule", "Effect"]),
                         use_container_width=True, hide_index=True)

        st.subheader("3. ε-NFA graph")
        st.graphviz_chart(to_dot(nfa), use_container_width=True)
        st.caption("Double circle = accept state, dashed edge = ε-transition.")

        st.subheader("4. Transition table δ")
        st.dataframe(transition_table(nfa), use_container_width=True, hide_index=True)

        st.subheader("5. Test strings (valid / invalid inputs)")
        w = st.text_input("Input string (blank = ε)", value="", key="w")
        bad = [c for c in w if c not in nfa.alphabet]
        if st.button("Simulate"):
            ok, trace = simulate(nfa, w)
            if bad:
                st.warning(f"Symbols {sorted(set(bad))} are not in the alphabet — string will be rejected.")
            (st.success if ok else st.error)(f"\"{w or EPS}\" → {'ACCEPTED ✓' if ok else 'REJECTED ✗'}")
            st.dataframe(pd.DataFrame(
                [{"Read": r, "ε-closure state set": fmt(s) + (f"  ∋ q{nfa.accept}" if nfa.accept in s else "")}
                 for r, s in trace]), use_container_width=True, hide_index=True)

        if st.button("Run sample tests"):
            tests = SAMPLES.get(regex.strip(), ["", "a", "b", "ab", "ba", "aab", "abb"])
            st.dataframe(pd.DataFrame(
                [{"Input": t or EPS, "Result": "Accepted ✓" if simulate(nfa, t)[0] else "Rejected ✗"}
                 for t in tests]), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Regex validation suite (valid & invalid expressions)")
    rows = []
    for r in SUITE:
        try:
            m = compile_regex(r)
            rows.append({"Regex": r or "(empty)", "Verdict": "Valid", "States": m.n_states, "Details": m.postfix})
        except RegexError as e:
            rows.append({"Regex": r or "(empty)", "Verdict": "Invalid", "States": "–", "Details": e.msg})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
