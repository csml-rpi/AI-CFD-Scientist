#!/bin/bash
# Changes OpenFOAM v1706 (2017) needs to compile with a current gcc. None
# changes what OpenFOAM computes. Each is an exact text replacement, applied
# only where the original text is still there, so running this twice is safe.
#
#   patches_v1706.sh <OpenFOAM-v1706 dir> <ThirdParty-v1706 dir> [<hyStrath dir>]
set -e
OF="$1"
HY="${3:-}"

python3 - "$OF" "$HY" <<'EOF'
import sys
from pathlib import Path

of = Path(sys.argv[1])
hy = Path(sys.argv[2]) if sys.argv[2] else None
PATCHES = [
    # A void operator= that ends in `return *this;`. gcc 11 rejects it when
    # the template is parsed; v1712 dropped the line.
    ("src/OpenFOAM/containers/Lists/PackedList/PackedListI.H",
     """    // Avoid going past end()
    // eg, iter = iterator(list, Inf)
    if (this->index_ > this->list_->size_)
    {
        this->index_ = this->list_->size_;
    }

    return *this;
}


template<unsigned nBits>
inline typename Foam::PackedList<nBits>::iterator&
Foam::PackedList<nBits>::iterator::operator++()""",
     """    // Avoid going past end()
    // eg, iter = iterator(list, Inf)
    if (this->index_ > this->list_->size_)
    {
        this->index_ = this->list_->size_;
    }
}


template<unsigned nBits>
inline typename Foam::PackedList<nBits>::iterator&
Foam::PackedList<nBits>::iterator::operator++()"""),
]
# PackedBoolList derives (through PackedList) from a private List<unsigned
# int>, so inside it a bare `UList` names that private base, and gcc 11
# refuses `UList<bool>`. Spelled Foam::UList, as later releases do.
for rel in ("PackedBoolList.H", "PackedBoolListI.H", "PackedBoolList.C"):
    PATCHES.append(("src/OpenFOAM/containers/Lists/PackedList/" + rel,
                    "(const UList<bool>& lst)", "(const Foam::UList<bool>& lst)"))
# Reaction keeps its name in a `const word name_` yet has a non-const name()
# returning word&; gcc 11 rejects binding the reference. The member loses its
# const, as in later releases; nothing assigns to it. hyStrath's Reaction2
# copies the same pattern.
PATCHES.append(("src/thermophysicalModels/specie/reaction/Reactions/Reaction/Reaction.H",
                "        const word name_;", "        word name_;"))
HY_PATCHES = [("src/thermophysicalModels/strath/strathSpecie/reaction/Reactions/Reaction/Reaction2.H",
               "        const word name_;", "        word name_;")]
# rampInlet's amplitude() and timeDuration() return scalar from members that
# are autoPtr<Function1<scalar>>; nothing calls them, and gcc 11 rejects the
# conversion when the template is parsed. They return the members' own type.
RAMP = "src/finiteVolume/fields/fvPatchFields/derived/rampInlet/rampInletFvPatchField.H"
for name in ("amplitude", "timeDuration"):
    HY_PATCHES.append((RAMP, f"""            scalar {name}() const
            {{
                return {name}_;
            }}""", f"""            const autoPtr<Function1<scalar> >& {name}() const
            {{
                return {name}_;
            }}"""))
    HY_PATCHES.append((RAMP, f"""            scalar& {name}()
            {{
                return {name}_;
            }}""", f"""            autoPtr<Function1<scalar> >& {name}()
            {{
                return {name}_;
            }}"""))
todo = [(of, rel, old, new) for rel, old, new in PATCHES]
if hy is not None:
    todo += [(hy, rel, old, new) for rel, old, new in HY_PATCHES]
for root, rel, old, new in todo:
    path = root / rel
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new))
        print(f"patched {rel}")
    elif new in text:
        print(f"already patched {rel}")
    else:
        sys.exit(f"patch target not found in {rel}")
EOF
