/*
  EXAMPLE 2: Use ContextChatENgine, which remembers the messages in the current conversation
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

  // const storageContext = await storageContextFromDefaults({
  //   persistDir: './storage',
  // });

  const index = await VectorStoreIndex.fromDocuments([
    ...tulaiDocuments,
    ...abramovDocuments,
    ...resumeDocuments
  ]);

  // const index = await VectorStoreIndex.init({
  //   storageContext
  // });

  const retriever = index.asRetriever();

  const chatEngine = new ContextChatEngine({
    retriever,
  });

  const rl = readline.createInterface({ input, output });

  while (true) {
    const query = await rl.question("Query: ");
    // const stream = await chatEngine.chat({ message: query, stream: true });
    // console.log();
    // for await (const chunk of stream) {
    //   process.stdout.write(chunk.message.content as string);
    // }
    const response = await chatEngine.chat({ message: query });
    console.log(response);
    // process.stdout.write(response.message.content as string);
  }
}

main().catch(console.error);