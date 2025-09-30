import re
import sys
from collections import defaultdict

def classify_task(task, client, model):
    """
    Map original task + client to new task type.
    """
    if task == "image_generation":
        if model == 'FLAVA':
            if str(client).endswith(("3", "9")):
                return "caltech_image_generation"
            elif str(client).endswith("8"):
                return "fmnist_image_generation"
        elif model == 'ViLT':
            if str(client).endswith(("0", "7", "8")):
                return "fmnist_image_generation"
            elif str(client).endswith(("3", "9")):
                return "caltech_image_generation"
    elif task == "text_generation":
        if str(client).endswith(("4", "5")):
            return "vqa_text_generation"
        elif str(client).endswith("6"):
            return "mmlu_text_generation"
        elif str(client).endswith(("8", "9")):
            return "commongen_text_generation"

    #keep original
    return task

def parse_out_file(file_path, model):
    """
    Parse log file and compute cumulative replacement counts per task type.
    """
    counts_over_time = defaultdict(lambda: [0] * 10)
    current_clients = []
    current_round = -1
    current_client_idx = -1  # index into current_clients

    with open(file_path, "r") as f:
        for line in f:
            #detect round header
            round_match = re.match(r"=== Round (\d+)/\d+, Clients: \[(.*?)\] ===", line)
            if round_match:
                current_round = int(round_match.group(1))
                clients_str = round_match.group(2)
                current_clients = [int(c.strip()) for c in clients_str.split(",")]
                current_client_idx = -1 
                continue

            #detect training progress bar, which determines which client is active
            train_match = re.search(r"Training clients:\s+(\d+)%\|.*\s(\d+)/(\d+)", line)
            if train_match and current_clients:
                current_client_idx += 1

            #detect replacement event
            repl_match = re.match(r"for task (\w+), replacement has taken place!", line)
            if repl_match and current_round >= 0 and current_client_idx is not None:
                task = repl_match.group(1)
                client = current_clients[current_client_idx]
                mapped_task = classify_task(task, client, model)
                bin_index = min(current_round // 20, 9)  # 0–19 -> 0, 20–39 -> 1, ...
                for i in range(bin_index, 10):
                    counts_over_time[mapped_task][i] += 1

    return counts_over_time

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python parse_out.py <path_to_out_file> <model>")
        sys.exit(1)

    file_path = sys.argv[1]
    model = sys.argv[2]
    counts_over_time = parse_out_file(file_path, model)

    print("Cumulative replacement counts per task type (at rounds 20,40,...,200):")
    for task, counts in counts_over_time.items():
        print(f"{task}: {counts}")