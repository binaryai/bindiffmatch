import argparse
import dataclasses

from .bindiffmatch import do_bindiffmatch
from .models import BinaryFile, MatchResult
from .utils import dump_matchresult, load_doc


@dataclasses.dataclass
class BindiffMatchConfig:
    threshold_high: float
    threshold_low: float
    threshold_remain: float
    hop: int


defaultconfig = BindiffMatchConfig(
    threshold_high=0.74,
    threshold_low=0.4,
    threshold_remain=0.66,
    hop=1,
)


def bindiffmatch(doc1: BinaryFile, doc2: BinaryFile, *, config: BindiffMatchConfig) -> MatchResult:
    return do_bindiffmatch(doc1, doc2, **dataclasses.asdict(config))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("-o", "--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    first = args.first
    second = args.second
    output = args.output
    doc1 = load_doc(first)
    doc2 = load_doc(second)
    matchresult = do_bindiffmatch(doc1, doc2, **dataclasses.asdict(defaultconfig))
    dump_matchresult(matchresult, output)


if __name__ == "__main__":
    main()
