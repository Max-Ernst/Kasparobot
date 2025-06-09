# MEET KASPAROBOT

Kasparobot is a little game I made for myself in partnership with a friend. Our original goal was to create a chess playing robot. While my friend is handling the mechanics end, I would handle the code end.
### Robotic control code hopefully coming soon! ###
With Kasparobot, you can play against a friend or against stockfish in the GUI or on command line.
I've also included a model for detecting chess pieces on a board, and the capability to translate those detections to the game board!

## Features

- Play the worlds best board game against an ai or a friend
- Detect chess pieces and board state from images using YOLO and custom mapping
- Uses the Stockfish engine for strong AI play or move suggestions

## Requirements

- Python 3.8+
- [matplotlib](https://matplotlib.org/)
- [python-chess](https://python-chess.readthedocs.io/)
- [cairosvg](https://cairosvg.org/)
- [ultralytics](https://docs.ultralytics.com/) (for YOLO)
- [stockfish](https://stockfishchess.org/) (binary included or specify path)

Install dependencies with:
```bash
pip install matplotlib python-chess cairosvg ultralytics
```

## Usage

1. **Clone the repository and navigate to the project directory.**
2. **Ensure you have requirements installed and stockfish binary downloaded.**
3. **Run the main program:**
   ```bash
   python main.py
   ```
4. **Choose your game mode:**
   - `ai` for playing against Stockfish
   - `pvp` for player vs player
5. **Interact with the game:**
   - Type commands (e.g., `move e4`, `get legal moves`, `show`, `get best move`, `reset`, `exit`)
   - Or click on the board to select and move pieces (in AI mode)

## Notes
- The project uses a custom-trained YOLO model for piece detection.
- All possible game states may not yet be accounted for - its hard to test on my own when I can't beat stockfish!
- This is a **WORK IN PROGRESS**, so not everything is integrated. The ability to play in CL and GUI was added for fun! Soon, this will all be handled physically with movement commands sent to the robotic arm based off of camera detections in real time!

## License
MIT License
