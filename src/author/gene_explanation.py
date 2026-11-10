from autogen import ConversableAgent

agent_explanation = ConversableAgent(
    name="agent_explanation",
    llm_config=llm_config,
    system_message="You are an interpreter, and you will receive a record of a student's work in json format that contains the content of the exercise, the answer, and whether it was correct or not. "
                   "You need to provide a short explanation for each record based on this information, explaining why the student answered the question incorrectly or correctly."
                   "The only thing you return is the json text containing the explanation, don't use snippet markup.",
    human_input_mode="NEVER",
)
