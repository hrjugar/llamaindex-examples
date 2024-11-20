/*
  EXAMPLE 3 PART 2: Feeding chat engine with previous past history
*/

import { stdin as input, stdout as output } from "node:process";
import readline from "node:readline/promises";

import {
  ChatMessage,
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

  const previousChatHistory: ChatMessage[] = [
    {
      content: "Who is Hernan?",
      role: "user",
    },
    {
      role: "assistant",
      content: "Hernan Rey N. Jugar appears to be an individual whose resume and another document (tulai.pdf) are being analyzed. \n\nFrom the provided context, we can infer that:\n\n1. Hernan has a Bachelor of Science in Computer Science degree from Mapúa Malayan Colleges Mindanao.\n2. He worked as a Web Development Intern at Tactiv Studios, where he led front-end development and collaborated on debugging code for Angular and Django projects.\n3. Hernan has experience working with various tools and frameworks, including React, Next.js, Figma, and MySQL.\n4. He participated in several extra-curricular activities, such as competitive programming contests and summer camp programs.\n\nThe tulai.pdf document seems to be a proposal or project description related to TulAi, an AI platform designed for remote areas with limited internet access. However, there is no explicit mention of Hernan's involvement in this project.",
    },
    {
      content: "What is tulai.pdf?",
      role: "user",
    },
    {
      role: "assistant",
      content: "Based on the provided context information, tulai.pdf appears to be a PDF document (file) located at the path `/Users/hernanjugar/Dev/Test/llamaindex-test/tulai.pdf`.\n\nThe content of this PDF file includes a proposal or project description related to TulAi, an AI platform designed for remote areas with limited internet access. The document is divided into sections, including:\n\n1. Executive Summary\n2. Introduction\n3. Product Overview (with features such as pre-built templates, support for Large Language Models, machine learning capabilities, and multilingual support)\n4. Market Need and Opportunity\n5. Business Model\n6. Impact and Benefits\n7. Conclusion\n\nThe PDF file is likely a report or proposal written by Hernan Jugar, given the file path and context information provided.",
    }
  ];

  const chatEngine = new ContextChatEngine({
    retriever,
    chatHistory: previousChatHistory
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
    // process.stdout.write(response.message.content as string);
  }

  console.log("--- FINISHED ---");
  console.log("--- FINISHED ---");
  console.log("--- FINISHED ---");
  console.log("CHAT HISTORY:");

  const chatHistory = await chatEngine.chatHistory;

  chatHistory.forEach((chat, i) => {
    console.log(`MESSAGE ${i}:`);
    console.log(JSON.stringify(chat, null, 2));
  })
}

main().catch(console.error);