import math
import os
import time
import threading
from collections import defaultdict, Counter
import cv2
import numpy as np
import mss
import keyboard
import pyautogui
# make it write the statistically best first word automatically (Tarse)
# as it takes the longest to calculate and is effectively constant reduces time to beat by 51 seconds
# Load word lists
with open("valid-wordle-words.txt", "r", encoding="utf-8") as f:
    all_words = [line.strip().lower() for line in f if len(line.strip()) == 5 and line.strip().isalpha()]

with open("wordle-answers-alphabetical.txt", "r", encoding="utf-8") as f:
    solution_words = [line.strip().lower() for line in f if len(line.strip()) == 5 and line.strip().isalpha()]

current_solutions = solution_words[:]  # current solutions
word_count = 1
lock = threading.Lock()

# BGR colour (update these with f8)
green  = (81, 132, 85)
yellow = (65, 140, 156)
grey   = (67, 65, 65)
#globals
bbox_map = {}
coordinate_set = False
corner1 = None
corner2 = None

# Coordinate setup

def capture_top_left():
    global corner1
    corner1 = pyautogui.position()
    print(f"Top-left corner set at: {corner1}")


def capture_bottom_right():
    global corner2, coordinate_set
    corner2 = pyautogui.position()
    print(f"Bottom-right corner set at: {corner2}")

    if corner1 is None:
        print("Please set the top-left corner first with 1.")
        return

    generate_bbox_map_from_corners()
    coordinate_set = True
    print("Bounding boxes generated.")
    print("Press 3 to make a guess\nPress 5 to exit")


def generate_bbox_map_from_corners():
    global bbox_map
    x1, y1 = corner1
    x2, y2 = corner2

    grid_width = x2 - x1
    grid_height = y2 - y1
    tile_width = grid_width // 5
    tile_height = grid_height // 6

    bbox_map.clear()
    for row in range(6):
        top = y1 + row * tile_height
        bbox_map[row + 1] = {
            "top": top,
            "left": x1,
            "width": tile_width * 5,
            "height": tile_height
        }

# Colour classification
def color_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

def classify_color(color):
    distances = {
        "green": color_distance(color, green),
        "yellow": color_distance(color, yellow),
        "grey": color_distance(color, grey)
    }
    return min(distances, key=distances.get)

# Interpret screen
def read_feedback_from_screen(row):
    bbox = bbox_map.get(row)
    if not bbox:
        return []

    tile_width = bbox["width"] // 5
    with mss.mss() as sct:
        screenshot = sct.grab(bbox)
        img = np.array(screenshot)
        feedback = []

        for i in range(5):
            x_start = i * tile_width
            tile = img[:, x_start:x_start + tile_width]

            if tile.shape[2] == 4:
                tile = cv2.cvtColor(tile, cv2.COLOR_BGRA2BGR)

            avg_color = tuple(np.mean(tile, axis=(0, 1)).astype(int))
            feedback.append(classify_color(avg_color))

        return feedback

# Wordle logical feedback for two words
def wordle_feedback(guess, solution):
    guess = list(guess)
    sol = list(solution)
    result = ['grey'] * 5

    # Green first
    for i in range(5):
        if guess[i] == sol[i]:
            result[i] = 'green'
            sol[i] = None
            guess[i] = None

    # Yellow second
    for i in range(5):
        if guess[i] is not None and guess[i] in sol:
            result[i] = 'yellow'
            sol[sol.index(guess[i])] = None

    return result

# Filter candidate solutions based on correct letters
def filter_words(word_list, guess, feedback):
    filtered = []
    required_counts = Counter()
    for i, (g_letter, result) in enumerate(zip(guess, feedback)):
        if result in ("green", "yellow"):
            required_counts[g_letter] += 1

    for word in word_list:
        valid = True
        word_letter_counts = Counter(word)

        # Green positions
        for i, (g_letter, result) in enumerate(zip(guess, feedback)): #removes possible words that have dont have this letter in the correct spot
            if result == "green" and word[i] != g_letter:
                valid = False
                break
        if not valid:
            continue

        # Yellow/grey checks
        for i, (g_letter, result) in enumerate(zip(guess, feedback)): #removes possible words that have this letter but in the same spot
            if result == "yellow":
                if g_letter == word[i] or g_letter not in word:
                    valid = False
                    break
            elif result == "grey": #removes possible words that contain these letters
                if word_letter_counts[g_letter] > required_counts[g_letter]:
                    valid = False
                    break

        if valid:
            filtered.append(word)

    return filtered

# Entropy calculations
def entropy_for_guess(guess, candidates): #For each word find all the possible green, yellow and grey letters for every possible solution
    patterns = defaultdict(int)
    for sol in candidates:
        pat = tuple(wordle_feedback(guess, sol))
        patterns[pat] += 1 #total number of patterns

    total = len(candidates) #total number of solutions
    ent = 0.0
    for count in patterns.values():
        p = count / total
        ent -= p * math.log2(p) #information entropy calculation
    return ent

def get_best_word_by_entropy(guess_list, candidates): #For every possible word find the amount of possibly outcomes of green yellow and grey letters
    best_word = None
    best_score = -1.0
    for g in guess_list:
        s = entropy_for_guess(g, candidates)
        if s > best_score:
            best_score = s
            best_word = g
    return best_word, best_score

# ----- Frequency-based fallback for large candidate sets -----
def get_best_word_by_frequency(word_list, top_k_letters=5):
    letter_counts = defaultdict(int)
    for word in word_list:
        for letter in set(word):
            letter_counts[letter] += 1
    sorted_letters = [l for l, _ in sorted(letter_counts.items(), key=lambda x: -x[1])]
    top_letters = set(sorted_letters[:top_k_letters])
    return max(word_list, key=lambda w: sum(1 for l in set(w) if l in top_letters))

# guess selector
def get_best_guess(candidates, all_words, solution_words, threshold=10):
    if len(candidates) <= threshold:
        return get_best_word_by_entropy(candidates, candidates) #when the possible solutions is less than 10 it only uses words that can be the solution
    else:
        return get_best_word_by_entropy(all_words, candidates) #uses every word as a guess even if it cannot be the answer to maximise the information a guess can provide

# typer function
def typer():
    global word_count, current_solutions

    with lock:
        if word_count > 6:
            print("Game over: 6 guesses used.")
            os._exit(0)
        print('Calculating best guess...')
        if word_count == 1:
            best_word = 'tarse' #automatically selects tarse as best word as it is constant
            best_entropy = entropy_for_guess('tarse', all_words)
        else:
            best_word, best_entropy = get_best_guess(current_solutions, all_words, solution_words, threshold=10)
        print(
            f"\nTyping: {best_word.upper()}"
            + (f"  (Entropy: {best_entropy:.3f} bits)" if best_entropy is not None else "")
        )
        keyboard.write(best_word + "\n")
        time.sleep(2) #waits for animation before looking at the screen

        feedback = read_feedback_from_screen(word_count)
        print(f"Feedback (row {word_count}): {feedback}")

        if feedback == ["green"] * 5:
            print(f"Wordle solved in {word_count} attempts! The word was: {best_word.upper()}")
            os._exit(0)

        if not feedback or len(feedback) != 5:
            print("Warning: couldn't read feedback reliably.")
            return

        current_solutions = filter_words(current_solutions, best_word, feedback)
        print(f"Words remaining: {len(current_solutions)}")
        if len(current_solutions) <= 40:
            print("Possible words:", current_solutions)

        word_count += 1

# debug view
def live_debug_view():
    global word_count
    while True:
        if not coordinate_set:
            time.sleep(0.1)
            continue

        bbox = bbox_map.get(word_count)
        if not bbox:
            continue

        tile_width = bbox["width"] // 5
        with mss.mss() as sct:
            screenshot = sct.grab(bbox)
            img = np.array(screenshot)

            for i in range(5):
                x_start = i * tile_width
                tile = img[:, x_start:x_start + tile_width]

                if len(tile.shape) == 3 and tile.shape[2] == 4:
                    tile = cv2.cvtColor(tile, cv2.COLOR_BGRA2BGR)

                avg_color = tuple(np.mean(tile, axis=(0, 1)).astype(int))
                label = classify_color(avg_color)

                cv2.rectangle(img, (x_start, 0), (x_start + tile_width, bbox["height"]), (0, 255, 0), 2)
                cv2.putText(img, label, (x_start + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.imshow("Live Tile View", img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()


#Log tile colours (calibration)
def log_tile_colors(row=1): #helper function to calibrate the tile colours
    bbox = bbox_map.get(row)
    if not bbox:
        print("Invalid row for logging.")
        return

    tile_width = bbox["width"] // 5
    with mss.mss() as sct:
        screenshot = sct.grab(bbox)
        img = np.array(screenshot)

        print(f"\nRow {row} tile BGR values:")
        for i in range(5):
            x_start = i * tile_width
            tile = img[:, x_start:x_start + tile_width]

            if tile.shape[2] == 4:
                tile = cv2.cvtColor(tile, cv2.COLOR_BGRA2BGR)

            avg_color = tuple(np.mean(tile, axis=(0, 1)).astype(int))
            print(f"Tile {i+1}: {avg_color}")

# Hotkeys and threads
debug_thread = threading.Thread(target=live_debug_view, daemon=True)
debug_thread.start()

keyboard.add_hotkey('1', capture_top_left)
keyboard.add_hotkey('2', capture_bottom_right)
keyboard.add_hotkey('3', typer)
keyboard.add_hotkey('4', lambda: log_tile_colors(row=word_count))
keyboard.add_hotkey('5', lambda: os._exit(0))
print('hello')
print("INSTRUCTIONS:\n"
      "1. Hover over the TOP-LEFT of the grid and press 1\n"
      "2. Hover over the BOTTOM-RIGHT of the grid and press 2\n"
      "3. Press 3 to make a guess\n"
      "4. Press 4 to log tile colors (for calibration)\n"
      "5. Press 5 to exit")
keyboard.wait()
