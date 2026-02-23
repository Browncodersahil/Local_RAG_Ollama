from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are a expert in answering quetion about a pizza restaurant

Here are some relevat reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n------------------------------")
    question = input("Ask your question(q to quit): ")
    print("\n\n")
    if question == "q":
        break

    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result) 






#Tkinter GUI try
# import tkinter as tk
# from tkinter import scrolledtext
# from langchain_ollama.llms import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate
# from vector import retriever

# model = OllamaLLM(model="llama3.2")

# template = """
# You are an expert in answering questions about a pizza restaurant.

# Here are some relevant reviews: {reviews}

# Here is the question to answer: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)
# chain = prompt | model

# def ask_question():
#     question = question_entry.get()
#     reviews = retriever.invoke(question)
#     result = chain.invoke({"reviews": reviews, "question": question})
#     result_text.delete(1.0, tk.END)  # Clear previous result
#     result_text.insert(tk.END, result)  # Insert new result

# # Create the main window
# window = tk.Tk()
# window.title("Pizza Restaurant Q&A")

# # Create and place the widgets
# question_label = tk.Label(window, text="Enter your question:")
# question_label.pack()

# question_entry = tk.Entry(window, width=50)
# question_entry.pack()

# ask_button = tk.Button(window, text="Ask", command=ask_question)
# ask_button.pack()

# result_text = scrolledtext.ScrolledText(window, width=60, height=20)
# result_text.pack()

# # Run the application
# window.mainloop()