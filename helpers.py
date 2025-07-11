import re, torch, openai, json, spacy
import numpy as np
from difflib import SequenceMatcher
from transformers import BertTokenizer, BertForSequenceClassification
from typing import Dict
from tqdm import tqdm


def extract_commit_parts(commit_message):
    """
    Extracts the prefix, description, and body from a commit message.

    Args:
        commit_message (str): The full commit message.

    Returns:
        dict: A dictionary containing:
            - "prefix": The extracted prefix (type only, without scope or colon).
            - "description": The message description after the prefix and colon.
            - "body": The body part of the commit message (everything after the first newline).
    """
    
    valid_prefix_pattern = r"^[a-zA-Z]+(\([\w\s,]+\))?(!)?$"
    
    # Replace escaped newlines
    commit_message = commit_message.replace("\\n", "\n")
    
    # Split first line and body
    first_line, _, body = commit_message.partition("\n")
    
    # Temporarily remove parentheses for colon detection
    first_line_no_parentheses = re.sub(r"\(.*?\)", "", first_line)
    
    colon_index = first_line_no_parentheses.find(":")
    if 0 < colon_index <= 10:
        potential_prefix = first_line[:colon_index].strip()

        normalized_prefix = re.sub(r"\s*\(\s*", "(", potential_prefix)
        normalized_prefix = re.sub(r"\s*\)\s*", ")", normalized_prefix)

        if re.match(valid_prefix_pattern, normalized_prefix):
            # Extract type only (before parentheses)
            type_only = normalized_prefix.split("(")[0]

            # New logic: find real colon index in original first_line
            real_colon_index = first_line.find(":")
            if real_colon_index != -1:
                description = first_line[real_colon_index + 1:].strip()
            else:
                description = first_line.strip()

            prefix = type_only
        else:
            prefix = None
            description = first_line.strip()
    else:
        prefix = None
        description = first_line.strip()

    return {
        "prefix": prefix,
        "description": description,
        "body": body.strip() if body else None
    }
    


def evaluate_prefix(prefix):
    """
    Evaluates whether a prefix is a valid Conventional Commit prefix and provides an error code,
    ignoring any scope present in parentheses and accounting for the `!` modifier.
    
    Args:
        prefix (str): The prefix to evaluate.
    
    Returns:
        dict: A dictionary containing:
            - "grade": A float between 0, 0.5, and 1.
            - "error_code": An integer representing the type of error:
                - 0: No error (perfect prefix).
                - 1: Uppercase error.
                - 2: Typo error.
                - 3: Both uppercase and typo errors.
                - 4: Not a valid prefix.
    """
    if not prefix:
        return {"grade": 0, "error_code": 4}  # No prefix provided or invalid format

    # List of valid Conventional Commit prefixes
    valid_prefixes = [
        "feat", "fix", "docs", "style", "refactor", "test",
        "chore", "perf", "ci", "build", "revert", "wip"
    ]

    # Strip out the scope (text within parentheses)
    prefix_cleaned = re.sub(r"\(.*?\)", "", prefix).strip()

    # Check for `!` at the end of the prefix and remove it for validation
    has_bang = prefix_cleaned.endswith("!")
    if has_bang:
        prefix_cleaned = prefix_cleaned[:-1]

    # Initialize error flags
    is_uppercase = prefix_cleaned.lower() != prefix_cleaned
    max_similarity = 0
    most_similar_prefix = None

    # Check similarity with valid prefixes
    for valid_prefix in valid_prefixes:
        similarity = SequenceMatcher(None, prefix_cleaned.lower(), valid_prefix).ratio()
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_prefix = valid_prefix

    # Apply thresholds for similarity
    if max_similarity < 0.8:
        return {"grade": 0, "error_code": 4}  # Too dissimilar to any valid prefix

    # Determine error codes and grades
    is_typo = prefix_cleaned.lower() != most_similar_prefix

    if is_uppercase and is_typo:
        return {"grade": 0, "error_code": 3}  # Both uppercase and typo errors
    elif is_uppercase:
        return {"grade": 0.5, "error_code": 1}  # Only uppercase issue
    elif is_typo:
        return {"grade": 0.5, "error_code": 2}  # Only typo issue

    # Perfect match
    return {"grade": 1, "error_code": 0}  # Perfect prefix



def grade_description(description, tokenizer, model):
    """
    Grades the description of a commit message using the trained BERT model.

    Args:
        commit_message (str): The full commit message to grade.
        tokenizer: tokenizer to be used (already loaded, not the tokenizer name)
        model: The BERT model to use (already loaded, not the model name)

    Returns:
        int: Predicted grade (0, 1, 2, or 3).
    """
    # Import the helper function to extract the description
    description = description

    if not description:
        raise ValueError("The commit message does not contain a valid description to evaluate.")

    # Prepare the input (use only the description part)
    inputs = tokenizer(description, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_grade = torch.argmax(logits, dim=1).item()

    return predicted_grade



def is_body_meaningful(description, body, no_openai=True):
    """
    Uses OpenAI API to evaluate if the body adds meaningful information to the description.

    Args:
        description (str): The description (title) of the commit message.
        body (str): The body of the commit message.
        no_openai: If True, function will return True by default if body is not None (without prompting GPT)

    Returns:
        bool: True if the body is meaningful, False otherwise.
    """
    if not body:  # No description provided
        return False

    if no_openai:
        return True

    # Construct the prompt
    prompt = (
        f"Evaluate the following commit message description and its body:\n\n"
        f"Description: {description}\n"
        f"Body: {body}\n\n"
        f"Does the body add meaningful information to the description? "
        f"Respond with 'TRUE' if it adds value, otherwise 'FALSE'."
    )

    try:
        # Query OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert commit message reviewer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,  # We only expect a short response
            temperature=0.2  # Lower temperature for deterministic output
        )

        # Extract the response content
        result = response.choices[0].message.content

        # Interpret the result
        if result == "TRUE":
            return True
        elif result == "FALSE":
            return False
        else:
            raise ValueError("Unexpected response from OpenAI API.")
    except Exception as e:
        print(f"Error evaluating description: {e}")
        return False



def is_desc_too_long(content):
    """
    Detects if a commit message content is too long (more than 50 characters excluding spaces).

    Args:
        content (str): The content of the commit message.

    Returns:
        bool: True if the content has more than 50 characters (excluding spaces), False otherwise.
    """
    # Remove spaces and count characters
    char_count = len(content.replace(" ", ""))
    return char_count > 50



def is_body_too_long(body):
    """
    Check if any line in the body exceeds 72 characters, excluding spaces,
    and considering escaped newlines (\\n).

    Args:
        body (str): The body of the commit message.

    Returns:
        bool: True if any line exceeds 72 non-space characters, False otherwise.
    """
    # Handle the case where body is None
    if body is None:
        return False

    # Replace escaped newlines with actual newlines
    processed_body = body.replace("\\n", "\n")

    # Split the body into lines and check the length of each line without spaces
    lines = processed_body.split("\n")
    return any(len(line.replace(" ", "")) > 72 for line in lines)
    


def is_uppercase(content):
    """
    Checks if the first letter of the commit message content is uppercase.

    Args:
        content (str): The content of the commit message.

    Returns:
        bool: True if the first letter is uppercase, False otherwise.
    """
    # Strip leading spaces and check the first character
    if content.strip():  # Ensure the content is not empty
        return content.strip()[0].isupper()
    return False  # Return False for empty content



def is_imp_verb(sentence, nlp):
    """
    Check if the first word of a sentence is a verb in imperative mood or matches known imperative verbs.

    Args:
        sentence (str): The sentence to analyze.
        nlp: The spacy model to use (already loaded, not the model name)

    Returns:
        bool: True if the first word is a verb in imperative mood, False otherwise.
    """

    # List of known imperative verbs to improve accuracy
    IMPERATIVE_VERBS = {"fix", "add", "update", "remove", "delete", "create", "reformat", "implement", "refactor", "format"}

    # Process the sentence with SpaCy
    doc = nlp(sentence.strip())

    # Ensure the sentence has tokens
    if not doc:
        return False

    # Check the first token
    first_token = doc[0]

    # Case 1: Check if the first word is a VERB in base form (imperative)
    if first_token.pos_ == "VERB" and first_token.tag_ == "VB":
        return True

    # Case 2: Check if the lemma matches a known imperative verb (but exclude past tense or inflected forms)
    if first_token.lemma_ in IMPERATIVE_VERBS and first_token.tag_ not in {"VBD", "VBN"}:
        if first_token.text.lower() != first_token.lemma_:  # Exclude past forms like "added"
            return False
        return True

    # Case 3: Check explicitly for problematic past tense forms
    if first_token.text.lower() in {"added", "removed", "deleted", "updated"}:
        return False

    # Case 4: Fallback check for exact matching of lowercase words in the imperative list
    if first_token.text.lower() in IMPERATIVE_VERBS and first_token.tag_ not in {"VBD", "VBN"}:
        return True

    return False



def grade_commit_message(commit_message, nlp, tokenizer, model, no_openai=True):
    """
    Grades a commit message based on its description, prefix, and body, including checks for imperative verbs.

    Args:
        commit_message (str): The commit message to grade.
        nlp: The spacy model to use (already loaded, not the model name)
        tokenizer: tokenizer to be used (already loaded, not the tokenizer name)
        model: The BERT model to use (already loaded, not the model name)
        no_openai: If True, function will return True by default if body is not None (without prompting GPT)

    Returns:
        tuple: Final grade (float) and booleans for the applied checks.
            - is_desc_too_long (bool): True if the description is too long.
            - is_uppercase (bool): True if the first letter of the description is uppercase.
            - is_not_imp_verb (bool): True if the description does not start with an imperative verb.
            - is_perfect_prefix (bool): True if the prefix is perfect (no errors).
            - is_uppercase_prefix (bool): True if the prefix has an uppercase error.
            - is_typo_prefix (bool): True if the prefix has a typo error.
            - is_uppercase_and_typo_prefix (bool): True if the prefix has both uppercase and typo errors.
            - is_invalid_prefix (bool): True if the prefix is invalid.
            - is_body_meaningful (bool): True if the body is meaningful relative to the description.
            - is_body_too_long (bool): True if the body has lines exceeding 72 characters.
            - is_body_evaluated (bool): True if the body has been evaluated, False otherwise.
    """
    # Step 1: Extract prefix, description, and body
    extracted = extract_commit_parts(commit_message)
    prefix = extracted["prefix"]
    description = extracted["description"]
    body = extracted["body"]

    print("Extracted Parts:")
    print("Prefix:", prefix)
    print("Description:", description)
    print("Body:", body)
    print("================================================================================================")

    # Step 2: Grade description
    description_grade = grade_description(description, tokenizer, model)

    # Initialize final grade with the description grade
    final_grade = description_grade
    if final_grade > 1:
        final_grade += 1

    # Initialize booleans
    is_desc_too_long_result = False
    is_uppercase_result = False
    is_not_imp_verb_result = False
    is_body_meaningful_result = False
    is_body_too_long_result = False
    is_perfect_prefix = False
    is_uppercase_prefix = False
    is_typo_prefix = False
    is_uppercase_and_typo_prefix = False
    is_invalid_prefix = False
    is_body_evaluated = False

    # Step 3: Handle SWENT-style commit (no prefix)
    if prefix is None:
        # Check if description is too long
        is_desc_too_long_result = is_desc_too_long(description)
        if is_desc_too_long_result:
            final_grade -= 0.5

        # Check if description starts with an imperative verb
        if not is_imp_verb(description, nlp):
            is_not_imp_verb_result = True
            final_grade -= 0.5

    # Step 4: Handle Conventional Commit (with prefix)
    else:
        # Evaluate the prefix
        prefix_evaluation = evaluate_prefix(prefix)
        prefix_grade = prefix_evaluation["grade"]
        error_code = prefix_evaluation["error_code"]
        final_grade += prefix_grade

        # Set error code booleans
        is_perfect_prefix = error_code == 0
        is_uppercase_prefix = error_code == 1
        is_typo_prefix = error_code == 2
        is_uppercase_and_typo_prefix = error_code == 3
        is_invalid_prefix = error_code == 4

        # Check if description is too long
        is_desc_too_long_result = is_desc_too_long(description)
        if is_desc_too_long_result:
            final_grade -= 0.5

        # Check if the first letter is uppercase
        is_uppercase_result = is_uppercase(description)
        if is_uppercase_result:
            final_grade -= 0.5

        # Check if description starts with an imperative verb
        if not is_imp_verb(description, nlp):
            is_not_imp_verb_result = True
            final_grade -= 0.5

    # Step 5: Evaluate the body if it is not None
    if body is not None:
        is_body_evaluated = True
        is_body_meaningful_result = is_body_meaningful(description, body, no_openai)
        is_body_too_long_result = is_body_too_long(body)

        # Modify the grade only if the description grade is 1 or 2
        if description_grade == 1:
            if is_body_meaningful_result:
                if not is_body_too_long_result:
                    final_grade += 1
                else:
                    final_grade += 0.5

        elif description_grade == 2:
            if is_body_meaningful_result and not is_body_too_long_result:
                final_grade += 0.5

    # Ensure final grade is non-negative and capped at 5
    final_grade = max(final_grade, 0)
    final_grade = min(final_grade, 5)

    return (
        description_grade,
        final_grade,
        is_desc_too_long_result,
        is_uppercase_result,
        is_not_imp_verb_result,
        is_perfect_prefix,
        is_uppercase_prefix,
        is_typo_prefix,
        is_uppercase_and_typo_prefix,
        is_invalid_prefix,
        is_body_meaningful_result,
        is_body_too_long_result,
        is_body_evaluated,
    )



def extract_commit_messages(json_path):
    """
    Extracts commit messages from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing objects with keys 'commit_message' and 'grade'.

    Returns:
        list: A list of commit messages.
    """
    try:
        with open(json_path, "r") as file:
            data = json.load(file)

        # Extract commit messages
        commit_messages = [entry["commit_message"] for entry in data if "commit_message" in entry]
        return commit_messages

    except Exception as e:
        print(f"Error processing the file: {str(e)}")
        return []



def extract_grades(json_path):
    """
    Extracts grades from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing objects with keys 'commit_message' and 'grade'.

    Returns:
        list: A list of grades.
    """
    try:
        with open(json_path, "r") as file:
            data = json.load(file)

        # Extract grades
        grades = [entry["grade"] for entry in data if "grade" in entry]
        return grades

    except Exception as e:
        print(f"Error processing the file: {str(e)}")
        return []



def actual_vs_pred(commit_messages, validation_grades, num_messages, nlp, tokenizer, model, no_openai=True):
    """
    Validates commit message grades by grading a subset of messages and comparing them with validation grades.
    Also calculates custom metrics and prints detailed information for flagged messages.

    Args:
        commit_messages (list): Array of commit messages.
        validation_grades (list): Array of grades for validation.
        num_messages (int): Number of commit messages to validate (from the beginning of the array).
        nlp: The spacy model to use (already loaded, not the model name)
        tokenizer: tokenizer to be used (already loaded, not the tokenizer name)
        model: The BERT model to use (already loaded, not the model name)
        no_openai: If True, function will return True by default if body is not None (without prompting GPT)

    Returns:
        dict: A dictionary containing:
            - "flagged_messages": List of flagged messages with grade mismatches and detailed results.
            - "custom_accuracy": The accuracy of predictions within the error margin.
            - "custom_f1": A custom F1-like metric based on continuous values.
    """

    predicted_grades = []
    actual_grades = []

    # Initialize progress bar
    with tqdm(total=min(num_messages, len(commit_messages)), desc="Validating Commit Messages", unit="msg") as pbar:
        for i in range(min(num_messages, len(commit_messages))):
            commit_message = commit_messages[i]
            validation_grade = validation_grades[i]
            
            # Extract the full grading details
            (
                original_description_grade,
                final_grade,
                is_desc_too_long,
                is_uppercase,
                is_not_imp_verb,
                is_perfect_prefix,
                is_uppercase_prefix,
                is_typo_prefix,
                is_uppercase_and_typo_prefix,
                is_invalid_prefix,
                is_body_meaningful,
                is_body_too_long,
                is_body_evaluated,
            ) = grade_commit_message(commit_message, nlp, tokenizer, model, no_openai)

            # Collect predictions and actual grades
            predicted_grades.append(final_grade)
            actual_grades.append(validation_grade)

            # Update progress bar
            pbar.update(1)

    # Custom accuracy: Fraction of predictions within the error margin
    within_margin = [abs(pred - actual) for pred, actual in zip(predicted_grades, actual_grades)]
    custom_accuracy = np.mean(within_margin)

    # Custom F1-like metric: Continuous version of F1 score
    max_grade = max(validation_grades)
    precision = [(1 - abs(pred - actual) / max_grade) for pred, actual in zip(predicted_grades, actual_grades)]
    recall = precision  # Recall equivalent in this continuous case
    custom_f1 = np.mean([2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)])

    return custom_accuracy, custom_f1, predicted_grades, actual_grades


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def generate_feedback(commit_message, final_grade, description_grade, errors):
    """
    Generates modular feedback for a commit message using OpenAI API.

    Args:
        commit_message (str): The commit message.
        final_grade (float): The final grade for the commit message.
        description_grade (int): The intermediate description grade (0-3 scale).
        errors (dict): Dictionary of error flags returned by `grade_commit_message`.

    Returns:
        str: Feedback generated by OpenAI API.
    """
    # Initialize the feedback prompt
    prompt = (
        f"Evaluate the following commit message and provide a critique focusing only on the identified issues. "
        f"If you encounter terms like 'B1', 'B2', 'B3', 'bootcamp', or 'SWENT', note that these are part of the course context, "
        f"and they do not require clarification or improvement.\n\n"
        f"Here is the commit message:\n\n"
        f"Commit Message: {commit_message}\n\n"
        f"The final grade for this commit message is {final_grade}/5. "
        f"Here are the issues that need to be addressed:\n\n"
    )
    
    # Track if there are any identified issues
    has_issues = False

    # Feedback for description grade
    if description_grade == 0:
        prompt += (
            "The commit message lacks clarity and sufficient detail. It does not adequately explain the changes made.\n"
        )
        has_issues = True
    elif description_grade == 1:
        prompt += (
            "The commit message is somewhat clear but lacks important details about the changes. "
            "It needs to be more specific and descriptive.\n"
        )
        has_issues = True
    elif description_grade == 2:
        prompt += "The commit message is clear and detailed enough but could be slightly improved in structure or wording.\n"
        has_issues = True

    # Feedback for specific errors
    if errors["is_desc_too_long"]:
        prompt += (
            "The commit message description is too long. Descriptions should be concise, ideally under 50 characters, to improve readability.\n"
        )
        has_issues = True

    if errors["is_uppercase"]:
        prompt += (
            "The description starts with an uppercase letter. Conventional commit messages typically start with a lowercase letter.\n"
        )
        has_issues = True

    if errors["is_not_imp_verb"]:
        prompt += (
            "The description does not start with an imperative verb (look at what comes after ':' and not the prefix to correct the student). Conventional commit messages should use action verbs like 'add', 'fix', or 'implement' to clearly state the intent.\n"
        )
        has_issues = True

    # Feedback for prefix issues
    if errors["is_uppercase_prefix"]:
        prompt += "The prefix contains uppercase letters, which is not standard. Prefixes should be lowercase.\n"
        has_issues = True
    if errors["is_typo_prefix"]:
        prompt += "The prefix contains a typo. Ensure the spelling matches conventional prefixes like 'feat' or 'fix'.\n"
        has_issues = True
    if errors["is_uppercase_and_typo_prefix"]:
        prompt += (
            "The prefix contains both uppercase letters and typos. It must adhere to conventional standards by fixing these issues.\n"
        )
        has_issues = True
    if errors["is_invalid_prefix"]:
        prompt += (
            "The prefix is invalid. Use valid prefixes such as 'feat', 'fix', or 'test' to correctly categorize the commit.\n"
        )
        has_issues = True

    # Feedback for body evaluation (if evaluated)
    if errors.get("is_body_evaluated"):
        if description_grade != 3:
            if errors["is_body_meaningful"]:
                prompt += "The body provides meaningful information and adds valuable context to the description.\n"
        if not errors["is_body_meaningful"]:
            prompt += "The body does not add meaningful information to the description. Additional context or justification for the changes is required.\n"
            has_issues = True
        if errors["is_body_too_long"]:
            prompt += (
                "The body contains lines that exceed 72 characters. Long lines should be split into shorter ones for better readability.\n"
            )
            has_issues = True

    # Add a message if there are no issues
    if not has_issues:
        if final_grade == 5:
            prompt += (
                "The commit message is perfect and adheres fully to conventional commit standards. "
                "No improvements are needed.\n"
            )
        elif final_grade == 4:
            prompt += (
                "The commit message follows SWENT-style standards and is clear and effective. "
                "While no improvements are necessary, you could consider adopting the conventional commit format in the future. "
                "This could help you align with widely used best practices for commit messages.\n"
            )

    # Conclude with direct critique
    prompt += (
        "\nProvide only the identified issues or good practices (like a good body) in a straightforward and concise manner without using bullet points."
    )

    # Use OpenAI API to generate the feedback
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful and constructive teaching assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=200,
        )
        feedback = response.choices[0].message.content
    except Exception as e:
        feedback = f"Error generating feedback: {str(e)}"

    return feedback



def process_commits_and_generate_feedback(input_json_path, output_json_path, model_path, spacy_nlp_name="en_core_web_sm", start=None, end=None, no_openai=True):
    """
    Processes commit messages, grades them, and generates feedback.

    Args:
        input_json_path (str): Path to the input JSON file.
        output_json_path (str): Path to save the output JSON file.
        start (int): Starting index of the range of commits to process (inclusive).
        end (int): Ending index of the range of commits to process (exclusive).

    Returns:
        list: List of processed commit messages with grades and feedback.
    """
    try:
        # Extract all commit messages using the given function
        all_commit_messages = extract_commit_messages(input_json_path)

        # Apply slicing for the specified range
        commit_messages_to_process = all_commit_messages[start:end]  # Handles range properly

        # Load the chosen spacy model
        nlp = spacy.load(spacy_nlp_name)
        
        # Load the trained BERT model using the model's path
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)

        # Initialize results list
        results = []

        # Process commit messages with a progress bar
        for commit_message in tqdm(commit_messages_to_process, desc="Processing Commit Messages", unit="msg"):
            # Run grading logic
            grade_results = grade_commit_message(commit_message, nlp, tokenizer, model, no_openai)
            description_grade = grade_results[0]
            final_grade = grade_results[1]
            errors = {
                "is_desc_too_long": grade_results[2],
                "is_uppercase": grade_results[3],
                "is_not_imp_verb": grade_results[4],
                "is_perfect_prefix": grade_results[5],
                "is_uppercase_prefix": grade_results[6],
                "is_typo_prefix": grade_results[7],
                "is_uppercase_and_typo_prefix": grade_results[8],
                "is_invalid_prefix": grade_results[9],
                "is_body_meaningful": grade_results[10],
                "is_body_too_long": grade_results[11],
                "is_body_evaluated": grade_results[12],
            }

            # Generate feedback
            feedback = generate_feedback(commit_message, final_grade, description_grade, errors)

            # Append result
            results.append({
                "commit_message": commit_message,
                "grade": final_grade,
                "feedback": feedback
            })

        # Save the results to the output JSON file
        with open(output_json_path, "w") as outfile:
            json.dump(results, outfile, indent=4)

        return results

    except Exception as e:
        print(f"Error: {e}")
        return []



def extract_commit_messages_per_sciper(json_path):
    """
    Extracts commit messages grouped by sciper ID from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing data.

    Returns:
        dict: A dictionary where keys are sciper IDs and values are lists of commit entries.
    """
    try:
        with open(json_path, "r") as file:
            data = json.load(file)

        # Group commits by sciper IDs
        commit_data = {}
        for sciper, commits in data.items():
            commit_data[sciper] = commits

        return commit_data

    except Exception as e:
        print(f"Error processing the file: {str(e)}")
        return {}