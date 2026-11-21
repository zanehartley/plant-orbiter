"""Fruit-count prompts for the text-to-image step (t2i_qwen.py --counts).

The fruit-counting experiment asks for N fruit, with the count written as a word ("five tomatoes", "one tomato"): Qwen-Image follows number
words better than digits. t2i_qwen.py then appends the framing sentence, e.g.
"A photo of tomato plants with five tomatoes. Single potted plant, whole plant and pot fully in frame ...".
"""
COUNT_PROMPT = "A photo of {species} plants with {count} {fruit}"
FRUIT = {"tomato": ("tomato", "tomatoes"), "lemon": ("lemon", "lemons")}  # singular (count 1), plural
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty"}


def number_word(n):
    """5 -> "five", 23 -> "twenty-three" (0-59)."""
    if n < 20:
        return ONES[n]
    tens, ones = TENS[n - n % 10], n % 10
    return f"{tens}-{ONES[ones]}" if ones else tens


def count_parts(species, count, words=False):
    """(number, fruit): ("5", "tomatoes"); ("five", "tomatoes") with words; count 1 takes the singular ("tomato")."""
    return number_word(count) if words else str(count), FRUIT[species][count != 1]


def count_prompt(species, count, words=False):
    """COUNT_PROMPT for one species and count, e.g. "A photo of tomato plants with five tomatoes"."""
    number, fruit = count_parts(species, count, words)
    return COUNT_PROMPT.format(species=species, count=number, fruit=fruit)
