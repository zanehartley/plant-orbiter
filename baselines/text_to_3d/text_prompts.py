"""Prompts and seeds for the text-to-3D baselines, shared by their run scripts (--experiment).

  synthetic_plants  PROMPT + ". " + OBJECT + ".", e.g. "A photo of bean plants. Single potted plant, whole plant."
                    PROMPT is PlantDreamer's guidance prompt, verbatim; OBJECT says what the target is. Species bean,
                    kale, mint x seeds 1-11 and 42; assets <species>_t<seed>.
  fruit_counting    count_prompt(species, N, words=True) + ". " + OBJECT + ".", e.g. "A photo of tomato plants with
                    five tomatoes. Single potted plant, whole plant." The count is a word, not digits. Species tomato,
                    lemon x N = 1-7 x seeds 42, 1, 2; assets <species>_n<count>_t<seed>.

Our text-to-image prompt (pipeline/t2i_qwen.py) is the same text with its final period replaced by FRAMING's
remainder (pot fully in frame, camera, background, lighting: instructions for a single 2D image). Each method adds
only its own fixed wrapper, if its inference code has one. Seeds are the same numbers as our text-to-image seeds;
the baselines share no input with our assets, so they are named <...>_t<seed> and compared unpaired.

The metrics score every method against PROMPT (synthetic plants) or COUNT_EVAL_PROMPT, the count prompt without its
number (fruit counting), so the quality scores do not re-measure the count.
"""
PROMPT = "A photo of {species} plants"
# Verbatim copy of pipeline/t2i_qwen.py's framing suffix (that module imports diffusers at load, so not imported here)
FRAMING = ("Single potted plant, whole plant and pot fully in frame and centred with margin around it, "
           "eye-level side view, isolated on a plain white background, studio lighting.")
OBJECT = "Single potted plant, whole plant"  # what the target is; the rest of FRAMING is how to photograph it
assert FRAMING.startswith(OBJECT + " ")  # so our text-to-image prompt = the baselines' prompt (minus its period) + suffix
COUNT_PROMPT = "A photo of {species} plants with {count} {fruit}"  # also in pipeline/prompts.py
FRUIT = {"tomato": ("tomato", "tomatoes"), "lemon": ("lemon", "lemons")}  # singular (count 1), plural
COUNT_EVAL_PROMPT = "A photo of {species} plants with {fruit}"  # fruit-counting metrics: COUNT_PROMPT without the number
EXPERIMENTS = ("synthetic_plants", "fruit_counting")
SPECIES = ("bean", "kale", "mint")
SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 42)
COUNT_SPECIES = ("tomato", "lemon")
COUNTS = tuple(range(1, 8))
COUNT_SEEDS = (42, 1, 2)
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


def count_eval_prompt(species):
    """COUNT_EVAL_PROMPT for one species, e.g. "A photo of tomato plants with tomatoes" (plural fruit, no number)."""
    return COUNT_EVAL_PROMPT.format(species=species, fruit=FRUIT[species][1])


def jobs(experiment):
    """(asset name, prompt, seed) for every asset of one experiment (EXPERIMENTS)."""
    if experiment == "fruit_counting":
        return [(f"{species}_n{count}_t{seed}", f"{count_prompt(species, count, words=True)}. {OBJECT}.", seed)
                for species in COUNT_SPECIES for count in COUNTS for seed in COUNT_SEEDS]
    if experiment == "synthetic_plants":
        return [(f"{species}_t{seed}", f"{PROMPT.format(species=species)}. {OBJECT}.", seed)
                for species in SPECIES for seed in SEEDS]
    raise ValueError(f"unknown experiment {experiment!r}; expected one of {EXPERIMENTS}")
