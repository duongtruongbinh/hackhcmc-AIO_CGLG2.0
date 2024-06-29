# -----***-----
# Prompt list for 5 business problems
# Business problem 1: Count the number of people using beer products
problem_1 = """
Analyze the given image and provide a detailed analysis that includes:
1. Identification of people:
- Identify and describe all individuals in the image.
- Clearly state the number of individuals identified and their specific activities and emotions.
2. Confirmation of the customers use beer products:
- Identify and describe all individuals holding or near a beer bottle, can, or glass in the image.
- Clearly state the number of individuals identified and their specific activities and emotions.
- Identify and describe all individuals holding or near a beer bottle, can, or glass in the image.
"""

# Business problem 2: Detect advertising or promotional items from beer brands
problem_2 = """
Analyze the given image to perform the following tasks:
1. Identify any logos present in the image:
- These logos may include text (with various typefaces/fonts), symbols, or a combination.
- Describe all items with the identified logo, providing details about the item's type, size, color, and appearance.
2. Identify advertisement or promotional items with identified logos in the image:
- Describe all advertisement or promotional items with the identified logo, such as refrigerators (or beverage coolers), advertising signs, posters, table standees, displays, standees, ice buckets, and parasols (if present).
Merge the same information and ignore duplicate information.
Comment on the overall presentation and organization of identified items.
"""

# Business problem 3: Evaluating the success of the event
problem_3 = """
Analyze the given image and provide a detailed analysis that includes:
1. Identification of people:
- Identify and describe all individuals in the image.
- Clearly state the number of individuals identified and their specific activities and emotions.
2. Confirmation of the customers use beer products:
- Clearly state the number of individuals individuals identified and their specific activities and emotions.
- Identify and describe all individuals holding or near a beer bottle, can, or glass in the image.
- Provide details about the beer product or nearby advertisement/marketing items including its appearance or brand logos (if present).
3. Crowd's emotion and activities recognition:
- Describe the overall activities and atmosphere of the crowd. Is it happy, angry, enjoyable, relaxed, neutral or something else?
"""

# Business problem 4: Track marketing staff
problem_4 = """
Analyze the given image to confirm the presence of marketing staff at the location. Provide a detailed analysis that includes:
1. Identification of Marketing Staff:
- Identify and describe all individuals wearing branding uniforms present in the image who are involved in marketing activities.
- Provide details on their appearance, clothing, logo (if present), and any visible branding or promotional materials they are handling.
2. Confirmation of Staff Presence:
- Clearly state the number of marketing staff members identified in the image and their specific activities related to product promotion.
- Verify whether there are at least 2 marketing staff members present at the location.
Ensure that the analysis is thorough and accurate, focusing on confirming the presence and activities of the marketing staff.
"""

# Business issue 5: Assess the level of presence of beer brands in convenience stores/supermarkets
problem_5 = """
Analyze the given image to perform the following tasks:
1. Identify any logos present in the image:
- The logos may include text (with various typefaces/fonts), symbols, or a combination.
- Describe all items with the identified logo, providing details about the item's type, size, color, and appearance.
2. Identify brand items and advertisement items with identified logos:
- Describe all packaging of brands, that have the identified logo.
- Describe all advertisement items with the identified logo, such as refrigerators (or beverage coolers), advertising signs, posters, table standees, standees, display stands, and parasols (if present).
Merge the same information and ignore duplicate information.
Comment on the overall presentation and organization of identified items in the store.
"""

default_opts = {"a photo at the convenience store": ["problem2", "problem4", "problem5"],
                "a photo at the supermarket": ["problem2", "problem4", "problem5"],
                "a photo at the bar or karaoke": ["problem2", "problem4"],
                "a photo at the event": ["problem2", "problem3", "problem4"],
                "a photo at the restaurant": ["problem1", "problem2", "problem3", "problem4"], }
# Define the business problem prompts
problem_prompts = {"problem1": problem_1,
                   "problem2": problem_2,
                   "problem3": problem_3,
                   "problem4": problem_4,
                   "problem5": problem_5}
