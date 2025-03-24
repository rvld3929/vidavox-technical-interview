# Documentation

#### Topic: Wakanda Country Statistical Report - The Richest Country Nobody Knows About

### **1. Generate Synthetic Dataset: Database/SQL**  

To generate a realistic synthetic dataset, a **ReAct Agent** was used with the **REPL Python** tool, allowing the LLM to utilize libraries such as **NumPy** and **Pandas**. The dataset is tailored to the given topic to ensure relevance and coherence.  

#### **Methodology**  
- **LLM Used:** `gemini-2.0-flash`  
- **Notebook:** `synthetic-dataset/synthetic-dataset.ipynb`  
- **Framework:** `LangChain` to build the ReAct Agent  
- **Generation Process:**  
  1. The **ReAct Agent** generates a **synthetic-dataset.csv** containing at least 1,000 rows of data with 10 features.  
  2. The CSV file is then converted into an **SQL format** (`synthetic-dataset.sql`).  
  3. Metadata describing the dataset structure and characteristics is stored in **metadata-synthetic-dataset.txt**.  
  4. This synthetic dataset is then used as a source for generating the **Synthetic PDF**, ensuring the document includes structured and relevant data.  

By following this approach, the synthetic dataset provides a solid foundation for creating a **realistic yet fabricated** document, making it suitable for **LLM-based text generation and structured document creation**.  

### **2. Generate Synthetic Dataset: PDF**  

To create a highly realistic synthetic PDF, the document generation process is done in multiple stages:  

1. **Outline Creation for Each Chapter**  
2. **Outline Creation for Each Subchapter within a Chapter**  
3. **Generating Paragraphs for Each Subchapter**  

#### **Implementation Details**  

- **Step 1 & 2 (Outline Generation):**  
  - A **LLM Chain** is used to generate structured outlines.  
  - The output is formatted using **JsonOutputParser** based on a **Pydantic** object to ensure consistency.  

- **Step 3 (Paragraph & Content Generation):**  
  - A **ReAct Agent** with **REPL Python** is used to generate detailed paragraphs.  
  - This approach allows for the dynamic creation of **charts and graphs** using **Matplotlib**.  

The entire process is implemented using **LangChain**, ensuring a structured, scalable, and flexible approach to generating synthetic documents.  

### **3. Create RAG System**  

#### **Chatbot Deployment Link**  
[Click Here](https://chatbot-deployment-vidavox-304986929596.us-central1.run.app)  

#### **Data Processing & Vectorization**  
1. **Splitting the PDF into Chunks**  
   - The synthetic PDF is first split into multiple **chunks** using **RecursiveCharacterTextSplitter** from LangChain.  
   - Chunking parameters:  
     - **Chunk size:** 100  
     - **Chunk overlap:** 100  
     - **Separator:** Ensures each chunk remains a complete paragraph.  

2. **Embedding & Vector Storage**  
   - The processed chunks are converted into **vector representations** using **Google’s text-embedding-004 model**.  
   - The generated embeddings are stored in the **Pinecone** vector database.  

#### **RAG Chatbot Implementation**  
- A **ReAct Agent** is used with a **retriever tool** to efficiently query the Pinecone vector database.  
- **LLM used:** `"qwen-2.5-32b"` from **Groq**.  
- **Conversation Memory:**  
  - A **ConversationSummaryMemory** is added to allow the chatbot to remember past interactions.  

#### **Deployment**  
- The chatbot system is deployed on **Google Cloud Run**.  
- The **user interface** is built using **Streamlit** for an interactive experience.  

##### **Running the Chatbot Locally**  
- To run the chatbot **locally**, use the following command in the terminal:  
  ```bash
  streamlit run app.py
  ```  
- Alternatively, the chatbot can be accessed directly via the provided deployment link.  

#### **Benchmark Questions**  
1. Apa faktor utama yang mendorong pertumbuhan ekonomi Wakanda dalam 1.000 hari terakhir?  
2. Bagaimana dampak vibranium terhadap PDB Wakanda?  
3. Bagaimana distribusi sektor tenaga kerja di Wakanda?  
4. Apa kebijakan Wakanda dalam menjaga keberlanjutan eksploitasi vibranium?  
5. Seberapa besar investasi Wakanda dalam teknologi, dan bagaimana dampaknya terhadap ekonomi?  
6. Bagaimana peran perempuan dalam struktur sosial dan ekonomi Wakanda?  
7. Apa hubungan antara tingkat pengangguran dan PDB di Wakanda?  
8. Bagaimana kebijakan perdagangan Wakanda memengaruhi ekspor berlian dan vibranium?  
9. Apa hubungan antara pengeluaran pemerintah untuk pendidikan dan kesehatan terhadap PDB?  
10. Bagaimana tren populasi Wakanda memengaruhi pasar tenaga kerja?  


### **4. Prepare/Format the Synthetic PDF Data for Fine-Tuning**  

The synthetic PDF is first divided into multiple **chunks** using the same chunking specifications as in the **RAG Chatbot** implementation. These chunks are then processed through an **LLM** to generate a dataset formatted for **fine-tuning an instruction model**.  

#### **Fine-Tuning Data Format**  
The generated dataset follows this structure and instruction:  
```json
{
    "instruction": "Each instruction must be clear, concise, and directly derived from the text chunk.",
    "input": "The input should include the provided text chunk only if necessary—otherwise, leave it empty",
    "output": "The output must be a well-structured and accurate response to the instruction."
}
```

#### **Implementation Details**  
- The **LLM** is used to generate structured JSON output.  
- **LangChain** is utilized with **JsonOutputParser**, ensuring adherence to a **Pydantic object** schema for consistency.  
