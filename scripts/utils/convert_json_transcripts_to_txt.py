import json
import os
import argparse
from pathlib import Path

def process_file(file_path: Path, is_speaker1_seller: bool, show_intervals: bool):
    """
    Reads a JSON transcript file and converts it to a formatted text file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if conversation and utterances exist
        if 'conversation' not in data or 'utterances' not in data['conversation']:
            print(f"Skipping {file_path.name}: 'conversation' or 'utterances' key missing.")
            return

        utterances = data['conversation']['utterances']
        
        # Output filename: replace .json with .txt
        output_path = file_path.with_suffix('.txt')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            previous_end_ms = None
            for utterance in utterances:
                start_ms = utterance.get('start_ms', 0)
                end_ms = utterance.get('end_ms', 0)
                
                # Insert interval if requested and not the first utterance
                if show_intervals and previous_end_ms is not None:
                    gap_ms = start_ms - previous_end_ms
                    gap_sec = gap_ms / 1000.0
                    f.write(f"[Pause: {gap_sec:.2f}s]\n\n")

                speaker_id = utterance.get('speaker_id')
                text = utterance.get('text', '')
                
                # Determine label
                if speaker_id == 1:
                    if is_speaker1_seller:
                        label = "Sælger"
                    else:
                        label = "Kunde"
                else:
                    if is_speaker1_seller:
                        label = "Kunde"
                    else:
                        label = "Sælger"
                
                f.write(f"{label}:\n\n{text}\n\n")
                
                previous_end_ms = end_ms
        
        print(f"Processed: {file_path.name} -> {output_path.name}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description="Convert JSON conversation transcripts to formatted text files.")
    parser.add_argument("input_file", type=str, help="Path to the JSON transcript file to convert.")
    
    # helper to handle both --flag and --flag=value styles if possible, but argparse doesn't support --flag value for store_true easily without type=bool which is tricky
    # The user specifically asked for: --is-speaker1-seller false
    # This implies type=some_parsing_function.
    
    parser.add_argument("--is-speaker1-seller", type=str2bool, nargs='?', const=True, default=False, 
                        help="Indicate if Speaker 1 is the Seller. Accepts 'true' or 'false'. Default is false (Speaker 1 is Customer).")
    
    parser.add_argument("--show-intervals", action="store_true", help="Flag to show time intervals (pauses) between utterances.")
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    
    if not input_file.exists():
        print(f"Error: File '{input_file}' does not exist.")
        return
    
    if not input_file.is_file():
        print(f"Error: '{input_file}' is not a file.")
        return
    
    # Process the single file
    print(f"Processing {input_file}...")
    process_file(input_file, args.is_speaker1_seller, args.show_intervals)
    print("Done.")

if __name__ == "__main__":
    main()
