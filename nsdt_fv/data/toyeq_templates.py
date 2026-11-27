"""The six ToyEq project templates of plan C.3 (list operations, arithmetic, trees, bit lists and
two that mix them) and the prelude. Each template: definitions (names without project prefix) and
about fifteen hand-written seed lemmas, ordered so that a seed uses only earlier seeds."""

ADD = """
def add : Nat Nat -> Nat
  add(Z, m) = m
  add(S(n), m) = S(add(n, m))
"""
APP = """
def app : List List -> List
  app(Nil, ys) = ys
  app(Cons(h, t), ys) = Cons(h, app(t, ys))
"""
LEN = """
def len : List -> Nat
  len(Nil) = Z
  len(Cons(h, t)) = S(len(t))
"""
REV = """
def rev : List -> List
  rev(Nil) = Nil
  rev(Cons(h, t)) = app(rev(t), Cons(h, Nil))
"""
DBL = """
def dbl : Nat -> Nat
  dbl(Z) = Z
  dbl(S(n)) = S(S(dbl(n)))
"""
MUL = """
def mul : Nat Nat -> Nat
  mul(Z, m) = Z
  mul(S(n), m) = add(m, mul(n, m))
"""
SUM = """
def sum : List -> Nat
  sum(Nil) = Z
  sum(Cons(h, t)) = add(h, sum(t))
"""
REP = """
def rep : Nat Nat -> List
  rep(Z, x) = Nil
  rep(S(n), x) = Cons(x, rep(n, x))
"""
TREE = """
def size : Tree -> Nat
  size(Leaf) = Z
  size(Node(l, v, r)) = S(add(size(l), size(r)))
def mirror : Tree -> Tree
  mirror(Leaf) = Leaf
  mirror(Node(l, v, r)) = Node(mirror(r), v, mirror(l))
def flat : Tree -> List
  flat(Leaf) = Nil
  flat(Node(l, v, r)) = app(flat(l), Cons(v, flat(r)))
def sumt : Tree -> Nat
  sumt(Leaf) = Z
  sumt(Node(l, v, r)) = add(sumt(l), add(v, sumt(r)))
def leaves : Tree -> Nat
  leaves(Leaf) = S(Z)
  leaves(Node(l, v, r)) = add(leaves(l), leaves(r))
def inct : Tree -> Tree
  inct(Leaf) = Leaf
  inct(Node(l, v, r)) = Node(inct(l), S(v), inct(r))
"""

ADD_SEEDS = [
    ("add_z_r", "forall n. add(n, Z) = n"),
    ("add_s_r", "forall n m. add(n, S(m)) = S(add(n, m))"),
    ("add_assoc", "forall a b c. add(add(a, b), c) = add(a, add(b, c))"),
    ("add_comm", "forall n m. add(n, m) = add(m, n)"),
]
APP_SEEDS = [
    ("app_nil_r", "forall xs. app(xs, Nil) = xs"),
    ("app_assoc", "forall xs ys zs. app(app(xs, ys), zs) = app(xs, app(ys, zs))"),
]

TEMPLATES = {
    "lists": {
        "defs": ADD + APP + LEN + REV + SUM + REP + """
def incs : List -> List
  incs(Nil) = Nil
  incs(Cons(h, t)) = Cons(S(h), incs(t))
def qrev : List List -> List
  qrev(Nil, acc) = acc
  qrev(Cons(h, t), acc) = qrev(t, Cons(h, acc))
""",
        "seeds": ADD_SEEDS + APP_SEEDS + [
            ("len_app", "forall xs ys. len(app(xs, ys)) = add(len(xs), len(ys))"),
            ("rev_app", "forall xs ys. rev(app(xs, ys)) = app(rev(ys), rev(xs))"),
            ("rev_rev", "forall xs. rev(rev(xs)) = xs"),
            ("len_rev", "forall xs. len(rev(xs)) = len(xs)"),
            ("sum_app", "forall xs ys. sum(app(xs, ys)) = add(sum(xs), sum(ys))"),
            ("len_incs", "forall xs. len(incs(xs)) = len(xs)"),
            ("incs_app", "forall xs ys. incs(app(xs, ys)) = app(incs(xs), incs(ys))"),
            ("qrev_app", "forall xs ys. qrev(xs, ys) = app(rev(xs), ys)"),
            ("len_rep", "forall n x. len(rep(n, x)) = n"),
            ("rev_incs", "forall xs. rev(incs(xs)) = incs(rev(xs))"),
            ("sum_rev", "forall xs. sum(rev(xs)) = sum(xs)"),
        ],
    },
    "arith": {
        "defs": ADD + MUL + DBL + """
def pred : Nat -> Nat
  pred(Z) = Z
  pred(S(n)) = n
def sub : Nat Nat -> Nat
  sub(n, Z) = n
  sub(n, S(m)) = pred(sub(n, m))
def neg : Bool -> Bool
  neg(T) = F
  neg(F) = T
def even : Nat -> Bool
  even(Z) = T
  even(S(n)) = neg(even(n))
def andb : Bool Bool -> Bool
  andb(T, b) = b
  andb(F, b) = F
def tri : Nat -> Nat
  tri(Z) = Z
  tri(S(n)) = S(add(n, tri(n)))
""",
        "seeds": ADD_SEEDS + [
            ("mul_z_r", "forall n. mul(n, Z) = Z"),
            ("mul_one", "forall n. mul(n, S(Z)) = n"),
            ("mul_add", "forall a b c. mul(add(a, b), c) = add(mul(a, c), mul(b, c))"),
            ("mul_assoc", "forall a b c. mul(mul(a, b), c) = mul(a, mul(b, c))"),
            ("dbl_add", "forall n. dbl(n) = add(n, n)"),
            ("neg_neg", "forall b. neg(neg(b)) = b"),
            ("even_dbl", "forall n. even(dbl(n)) = T"),
            ("sub_z_l", "forall n. sub(Z, n) = Z"),
            ("andb_t_r", "forall b. andb(b, T) = b"),
            ("andb_comm", "forall b c. andb(b, c) = andb(c, b)"),
            ("even_ss", "forall n. even(S(S(n))) = even(n)"),
            ("pred_add_s", "forall n m. pred(add(n, S(m))) = add(n, m)"),
        ],
    },
    "trees": {
        "defs": ADD + APP + LEN + REV + TREE,
        "seeds": ADD_SEEDS + APP_SEEDS + [
            ("len_app", "forall xs ys. len(app(xs, ys)) = add(len(xs), len(ys))"),
            ("rev_app", "forall xs ys. rev(app(xs, ys)) = app(rev(ys), rev(xs))"),
            ("mirror_mirror", "forall t. mirror(mirror(t)) = t"),
            ("size_mirror", "forall t. size(mirror(t)) = size(t)"),
            ("leaves_size", "forall t. leaves(t) = S(size(t))"),
            ("len_flat", "forall t. len(flat(t)) = size(t)"),
            ("flat_mirror", "forall t. flat(mirror(t)) = rev(flat(t))"),
            ("size_inct", "forall t. size(inct(t)) = size(t)"),
            ("mirror_inct", "forall t. mirror(inct(t)) = inct(mirror(t))"),
            ("leaves_mirror", "forall t. leaves(mirror(t)) = leaves(t)"),
            ("sumt_mirror", "forall t. sumt(mirror(t)) = sumt(t)"),
        ],
    },
    "bits": {
        "defs": ADD + APP + LEN + DBL + """
def flip : Nat -> Nat
  flip(Z) = S(Z)
  flip(S(n)) = Z
def one : Nat -> Nat
  one(Z) = Z
  one(S(n)) = S(Z)
def bnot : List -> List
  bnot(Nil) = Nil
  bnot(Cons(b, bs)) = Cons(flip(b), bnot(bs))
def pop : List -> Nat
  pop(Nil) = Z
  pop(Cons(b, bs)) = add(one(b), pop(bs))
def zeros : Nat -> List
  zeros(Z) = Nil
  zeros(S(n)) = Cons(Z, zeros(n))
def norm : List -> List
  norm(Nil) = Nil
  norm(Cons(b, bs)) = Cons(one(b), norm(bs))
def val : List -> Nat
  val(Nil) = Z
  val(Cons(b, bs)) = add(one(b), dbl(val(bs)))
""",
        "seeds": ADD_SEEDS + APP_SEEDS + [
            ("one_one", "forall n. one(one(n)) = one(n)"),
            ("flip_flip", "forall n. flip(flip(n)) = one(n)"),
            ("one_flip", "forall n. one(flip(n)) = flip(n)"),
            ("len_bnot", "forall bs. len(bnot(bs)) = len(bs)"),
            ("bnot_bnot", "forall bs. bnot(bnot(bs)) = norm(bs)"),
            ("pop_zeros", "forall n. pop(zeros(n)) = Z"),
            ("len_zeros", "forall n. len(zeros(n)) = n"),
            ("norm_norm", "forall bs. norm(norm(bs)) = norm(bs)"),
            ("pop_norm", "forall bs. pop(norm(bs)) = pop(bs)"),
            ("bnot_app", "forall xs ys. bnot(app(xs, ys)) = app(bnot(xs), bnot(ys))"),
            ("norm_app", "forall xs ys. norm(app(xs, ys)) = app(norm(xs), norm(ys))"),
            ("val_zeros", "forall n. val(zeros(n)) = Z"),
            ("val_norm", "forall bs. val(norm(bs)) = val(bs)"),
            ("pop_app", "forall xs ys. pop(app(xs, ys)) = add(pop(xs), pop(ys))"),
        ],
    },
    "mix_listarith": {
        "defs": ADD + MUL + DBL + APP + LEN + SUM + REP + """
def dbll : List -> List
  dbll(Nil) = Nil
  dbll(Cons(h, t)) = Cons(dbl(h), dbll(t))
def scale : List Nat -> List
  scale(Nil, k) = Nil
  scale(Cons(h, t), k) = Cons(mul(h, k), scale(t, k))
""",
        "seeds": ADD_SEEDS + APP_SEEDS + [
            ("len_app", "forall xs ys. len(app(xs, ys)) = add(len(xs), len(ys))"),
            ("sum_app", "forall xs ys. sum(app(xs, ys)) = add(sum(xs), sum(ys))"),
            ("dbl_add", "forall n m. dbl(add(n, m)) = add(dbl(n), dbl(m))"),
            ("len_dbll", "forall xs. len(dbll(xs)) = len(xs)"),
            ("sum_dbll", "forall xs. sum(dbll(xs)) = dbl(sum(xs))"),
            ("dbll_app", "forall xs ys. dbll(app(xs, ys)) = app(dbll(xs), dbll(ys))"),
            ("sum_rep", "forall n x. sum(rep(n, x)) = mul(n, x)"),
            ("len_rep", "forall n x. len(rep(n, x)) = n"),
            ("rep_add", "forall n m x. rep(add(n, m), x) = app(rep(n, x), rep(m, x))"),
            ("len_scale", "forall xs k. len(scale(xs, k)) = len(xs)"),
            ("scale_app", "forall xs ys k. scale(app(xs, ys), k) = app(scale(xs, k), scale(ys, k))"),
            ("mul_z_r", "forall n. mul(n, Z) = Z"),
            ("scale_z", "forall xs. sum(scale(xs, Z)) = Z"),
        ],
    },
    "mix_treelist": {
        "defs": ADD + APP + LEN + REV + SUM + TREE + """
def isleaf : Tree -> Bool
  isleaf(Leaf) = T
  isleaf(Node(l, v, r)) = F
def lsum : List -> List
  lsum(Nil) = Nil
  lsum(Cons(h, t)) = Cons(S(h), lsum(t))
""",
        "seeds": ADD_SEEDS + APP_SEEDS + [
            ("len_app", "forall xs ys. len(app(xs, ys)) = add(len(xs), len(ys))"),
            ("sum_app", "forall xs ys. sum(app(xs, ys)) = add(sum(xs), sum(ys))"),
            ("rev_app", "forall xs ys. rev(app(xs, ys)) = app(rev(ys), rev(xs))"),
            ("len_flat", "forall t. len(flat(t)) = size(t)"),
            ("sum_flat", "forall t. sum(flat(t)) = sumt(t)"),
            ("mirror_mirror", "forall t. mirror(mirror(t)) = t"),
            ("flat_mirror", "forall t. flat(mirror(t)) = rev(flat(t))"),
            ("isleaf_mirror", "forall t. isleaf(mirror(t)) = isleaf(t)"),
            ("lsum_app", "forall xs ys. lsum(app(xs, ys)) = app(lsum(xs), lsum(ys))"),
            ("len_lsum", "forall xs. len(lsum(xs)) = len(xs)"),
            ("size_mirror", "forall t. size(mirror(t)) = size(t)"),
            ("rev_rev", "forall xs. rev(rev(xs)) = xs"),
        ],
    },
}

# the prelude (base-model pretraining material, never in the stream) reuses the list template
PRELUDE_TEMPLATE = "lists"
STREAM_TEMPLATES = ["lists", "arith", "trees", "bits", "mix_listarith", "mix_treelist"]
