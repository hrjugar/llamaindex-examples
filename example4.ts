/*
  EXAMPLE 4: Inserting new documents to an existing storage context
*/
import fs from "node:fs/promises";

import {
  Document,
  MetadataMode,
  NodeWithScore,
  VectorStoreIndex,
  Settings,
  Ollama,
  HuggingFaceEmbedding,
  PDFReader,
  TextFileReader,
  storageContextFromDefaults
} from "llamaindex";

Settings.llm = new Ollama({
  model: "llama3.1",
  config: {}
});

Settings.embedModel = new HuggingFaceEmbedding({
  modelType: "BAAI/bge-small-en-v1.5",
});


async function main() {
  // const storageContext = await storageContextFromDefaults({
  //   persistDir: "./storage"
  // });

  // // const essay = await fs.readFile(path, "utf-8");
  const pdfReader = new PDFReader();
  const textFileReader = new TextFileReader();

  // const tulaiFilePath = "tulai.pdf";
  // const tulaiDocuments = await pdfReader.loadData(tulaiFilePath);

  // const abramovFilePath = "abramov.txt";
  // const abramovDocuments = await textFileReader.loadData(abramovFilePath);

  // const index = await VectorStoreIndex.fromDocuments([
  //   ...tulaiDocuments,
  //   ...abramovDocuments,
  // ], {
  //   storageContext
  // });

  // const queryEngine = index.asQueryEngine();

  // const response = await queryEngine.query({
  //   query: "What makes tulai special?",
  // });

  // // Output response
  // console.log(JSON.stringify(response, null, 2));

  // console.log("----------");
  // console.log("----------");
  // console.log("----------");
  // console.log("----------");
  // console.log("----------");

  const secondStorageContext = await storageContextFromDefaults({
    persistDir: './storage',
  });

  const loadedIndex = await VectorStoreIndex.init({
    storageContext: secondStorageContext
  });

  const resumeFilePath = "resume.pdf";
  const resumeDocuments = await pdfReader.loadData(resumeFilePath);

  const loadedQueryEngine = loadedIndex.asQueryEngine();

  for (const doc of resumeDocuments) {
    await loadedIndex.insert(doc);
  }

  console.log("INSERTION COMPLETE");

  const loadedResponse = await loadedQueryEngine.query({
    query: "Who is Hernan? Describe him."
  })

  console.log(JSON.stringify(loadedResponse, null, 2));
}

main().catch(console.error);