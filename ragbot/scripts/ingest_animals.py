import argparse
from datetime import datetime, timezone
from typing import List, Dict
from uuid import uuid4

from ragbot.api.elastic import ensure_index, index_docs
from ragbot.api.embeddings import embed_many


def _animal_names() -> List[str]:
    return [
        "African Elephant", "Asian Elephant", "Giraffe", "Zebra", "White Rhinoceros", "Black Rhinoceros",
        "Hippopotamus", "African Buffalo", "Bison", "Lion", "Tiger", "Leopard", "Jaguar", "Cheetah",
        "Cougar", "Lynx", "Bobcat", "Caracal", "Serval", "Domestic Cat", "Wolf", "Coyote",
        "Arctic Fox", "Red Fox", "Fennec Fox", "Brown Bear", "Polar Bear", "Panda", "Sloth Bear",
        "Koala", "Kangaroo", "Wallaby", "Wombat", "Platypus", "Echidna", "Camel", "Llama", "Alpaca",
        "Horse", "Donkey", "Goat", "Sheep", "Pig", "Cow", "Dingo", "Hyena", "African Wild Dog",
        "Crocodile", "Alligator", "Komodo Dragon", "Iguana", "Gecko", "Chameleon", "Tortoise",
        "Sea Turtle", "Python", "Cobra", "Viper", "Anaconda", "Bald Eagle", "Golden Eagle", "Hawk",
        "Falcon", "Owl", "Penguin", "Albatross", "Puffin", "Swan", "Duck", "Goose", "Turkey",
        "Chicken", "Peacock", "Ostrich", "Emu", "Cassowary", "Dolphin", "Orca", "Blue Whale",
        "Humpback Whale", "Manta Ray", "Stingray", "Great White Shark", "Hammerhead Shark",
        "Whale Shark", "Seal", "Sea Lion", "Walrus", "Octopus", "Squid", "Cuttlefish", "Crab",
        "Lobster", "Shrimp", "Jellyfish", "Starfish", "Clownfish", "Salmon", "Trout", "Tuna",
        "Parrot", "Macaw", "Toucan", "Lemur", "Gorilla", "Chimpanzee", "Bonobo", "Orangutan",
        "Baboon", "Mandrill", "Sloth", "Anteater", "Armadillo", "Pangolin", "Bat", "Beaver",
        "Otter", "Raccoon", "Skunk", "Porcupine", "Hedgehog", "Rabbit", "Hare", "Mouse", "Rat",
        "Squirrel", "Chipmunk", "Prairie Dog", "Meerkat", "Mongoose", "Civet", "Red Panda", "Giant Panda"
    ]


def _compose_content(name: str, idx: int) -> str:
    habitat_options = [
        "savannas", "tropical rainforests", "temperate forests", "grasslands", "mountains",
        "deserts", "wetlands", "coastal waters", "open ocean", "rivers and lakes",
    ]
    diet_options = [
        "herbivore", "carnivore", "omnivore", "insectivore", "piscivore",
    ]
    lifespan = ["~10 years", "~15 years", "~20 years", "~25 years", "~30+ years"][idx % 5]
    habitat = habitat_options[idx % len(habitat_options)]
    diet = diet_options[idx % len(diet_options)]
    return (
        f"{name} is commonly found in {habitat}. It is typically a {diet}. "
        f"Average lifespan in the wild is {lifespan}. This record is part of a synthetic knowledge base "
        f"for RAG retrieval demonstrations."
    )


def build_docs(count: int) -> List[Dict[str, object]]:
    names = _animal_names()
    if count > len(names):
        count = len(names)
    selected = names[:count]
    contents = [_compose_content(n, i) for i, n in enumerate(selected)]
    vectors = embed_many(contents)
    now = datetime.now(timezone.utc).isoformat()
    docs: List[Dict[str, object]] = []
    for i, (name, content, vec) in enumerate(zip(selected, contents, vectors)):
        docs.append({
            "id": str(uuid4()),
            "title": name,
            "url": "",
            "source": "animals",
            "content": content,
            "embedding": vec,
            "created_at": now,
        })
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest N animal records into Elasticsearch")
    parser.add_argument("--count", type=int, default=100, help="Number of records to ingest (max 120)")
    args = parser.parse_args()

    ensure_index()
    docs = build_docs(args.count)
    success, _ = index_docs(docs)
    print({"requested": args.count, "indexed": success})


if __name__ == "__main__":
    main()


