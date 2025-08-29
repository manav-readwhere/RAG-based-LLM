import argparse

from ragbot.api.ingest import ingest_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a directory of .md and .txt files")
    parser.add_argument("--path", required=True, help="Directory path to ingest")
    args = parser.parse_args()
    stats = ingest_directory(args.path)
    print(stats)


if __name__ == "__main__":
    main()


