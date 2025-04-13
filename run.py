 import argparse
from agent_routing import route_based_on_confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, help="The question to ask")
    parser.add_argument("--image", help="Optional image path")
    args = parser.parse_args()

    result, agent = route_based_on_confidence(args.question)
    print(f"[{agent}] {result}")
