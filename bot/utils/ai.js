const { OpenAI } = require("openai");
const { Pinecone } = require("@pinecone-database/pinecone");
const { encoding_for_model } = require("@dqbd/tiktoken");
const fs = require("fs");
const path = require("path");

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY});
const index = pinecone.Index(process.env.PINECONE_INDEX);
const encoder = encoding_for_model("gpt-4o");
const promptPath = path.join(__dirname, "../prompts/abeyan_prompt.txt");
const personalityPrompt = fs.readFileSync(promptPath, "utf-8");

function countTokens(req) {
  return req.reduce((total, msg) => {
    const content = msg.content || "";
    return total + encoder.encode(content).length;
  }, 0);
}

async function extractKeywords(prompt) {
  const systemPrompt = (
    `Extract < 5 short, relevant keywords or phrases that describe the user's intent. Avoid full sentences.` +
    `Respond only with a valid JSON array of keywords, e.g. ["cat", "dog", "fish"]. Do not add any extra text or formatting.`
  );
  const messages = [
    { role: "system", content: systemPrompt },
    { role: "user", content: `Prompt ${prompt}`}
  ]

  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages,
    temperature: 0.3
  });

  const content = completion.choices[0].message.content;
  const keywords = JSON.parse(content) || prompt;

  return keywords;
}

async function getAIResponse(prompt, contextHistory=[]) {
  const keywords = await extractKeywords(prompt);
  const inputs = [prompt, keywords.join(", ")].filter(s => s && s.trim().length > 0);
  let context = "";

  if (inputs.length !== 0) {
    const embeddings = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: inputs
    });
    
    const allResponses = []

    for (const emb of embeddings.data) {
      const results = await index.query({
        topK: 5,
        vector: emb.embedding,
        includeMetadata: true,
        filter: {
          author_names: { $in: ["Abeyan"]}
        }
      });

      for (const match of results.matches) {
        const msgCount = match.metadata?.message_count || 1;
        match.score = match.score * Math.log2(msgCount + 1);
        allResponses.push(match);
      }

      allResponses.push(...results.matches);
    }

    const seen = new Set();
    const deduped = allResponses
      .filter(match => {
        if (seen.has(match.id)) return false;
        seen.add(match.id);
        return true;
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);
    
    context = deduped
      .map((m) => `---\n${m.metadata.text}`)
      .join("\n\n")
  }

  const formattedHistory = contextHistory
    .map((turn) => `User: ${turn.prompt}\nYou: ${turn.response || ""}`)
    .join("\n");

  const systemPrompt = `${personalityPrompt}\n\nPrevious messages in this conversation:\n${formattedHistory}\n\nRetrieved context:\n${context}`
  
  const messages = [
      {
        role: "system",
        content: systemPrompt
      },
      {
        role: "user",
        content: prompt
      }
    ]

  const tokenCount = countTokens(messages);
  console.log(`${systemPrompt}\n\n`)
  console.log(`${tokenCount} tokens\nUser: ${prompt}}`)

  const chat = await openai.chat.completions.create({
    model: "gpt-4.1-mini",
    messages: messages,
    temperature: 0.1,
    top_p: 0.8
  });

  return chat.choices[0].message.content.trim();
}

module.exports = { getAIResponse };