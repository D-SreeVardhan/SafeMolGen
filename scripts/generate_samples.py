"""Generate SMILES samples from a trained generator."""

from pathlib import Path
import argparse

from models.generator.safemolgen import SafeMolGen
from utils.chemistry import validate_smiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/generator")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--valid-only", action="store_true")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--out", type=str, default="outputs/generator_samples.txt")
    args = parser.parse_args()

    model = SafeMolGen.from_pretrained(args.model)
    if args.valid_only:
        samples = model.generate_valid(
            n=args.n,
            temperature=args.temperature,
            top_k=args.top_k,
            max_attempts_per_sample=args.max_attempts,
            max_length=args.max_length,
        )
    else:
        samples = model.generate(
            n=args.n,
            temperature=args.temperature,
            top_k=args.top_k,
            max_length=args.max_length,
        )

    valid = [s for s in samples if validate_smiles(s)]
    unique = set(samples)
    valid_unique = set([s for s in valid])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Generated samples (n={len(samples)})\n")
        for s in samples:
            f.write(s + "\n")
        f.write("\nValid samples\n")
        for s in valid:
            f.write(s + "\n")

    print(f"Generated: {len(samples)}")
    print(f"Valid: {len(valid)} ({len(valid)/max(len(samples), 1):.2%})")
    print(f"Unique: {len(unique)} ({len(unique)/max(len(samples), 1):.2%})")
    print(f"Valid & unique: {len(valid_unique)} ({len(valid_unique)/max(len(samples), 1):.2%})")
    print(f"Saved samples to: {out_path}")

    print("\nSample valid SMILES:")
    for s in list(valid)[:10]:
        print("  ", s)

    print("\nSample generated (first 10):")
    for s in samples[:10]:
        print("  ", s)


if __name__ == "__main__":
    main()
