/*
  EXAMPLE 3: Acquiring chat history of current conversation
*/

import { stdin as input, stdout as output } from "node:process";
import readline from "node:readline/promises";

import {
  ContextChatEngine,
  Document,
  HuggingFaceEmbedding,
  Ollama,
  PDFReader,
  Settings,
  storageContextFromDefaults,
  TextFileReader,
  VectorStoreIndex,
} from "llamaindex";

Settings.llm = new Ollama({
  model: "llama3.1",
})

Settings.embedModel = new HuggingFaceEmbedding({
  modelType: "BAAI/bge-small-en-v1.5",
});

async function main() {
  const pdfReader = new PDFReader();
  const textFileReader = new TextFileReader();

  const tulaiFilePath = "tulai.pdf";
  const tulaiDocuments = await pdfReader.loadData(tulaiFilePath);

  const abramovFilePath = "abramov.txt";
  const abramovDocuments = await textFileReader.loadData(abramovFilePath);

  const resumeFilePath = "resume.pdf";
  const resumeDocuments = await pdfReader.loadData(resumeFilePath);

  const index = await VectorStoreIndex.fromDocuments([
    ...tulaiDocuments,
    ...abramovDocuments,
    ...resumeDocuments
  ]);

  const retriever = index.asRetriever();

  const chatEngine = new ContextChatEngine({
    retriever,
  });

  const rl = readline.createInterface({ input, output });

  let query = "";
  while (true) {
    query = await rl.question("Query: ");

    console.log(`provided query: ${query}`);

    if (query === "exit") {
      break;
    }

    const response = await chatEngine.chat({ message: query });
    console.log(JSON.stringify(response, null, 2));
  }

  console.log("--- FINISHED ---");
  console.log("--- FINISHED ---");
  console.log("--- FINISHED ---");
  console.log("CHAT HISTORY:");

  const chatHistory = await chatEngine.chatHistory;

  chatHistory.forEach((chat, i) => {
    console.log("CHAT 1:")
    console.log(JSON.stringify(chat, null, 2));
  })
}

main().catch(console.error);