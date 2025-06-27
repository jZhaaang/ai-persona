const { Events } = require('discord.js');
const { getAIResponse } = require("../utils/ai")

const MAX_CONTEXT_TURNS = 6;
const contextMap = new Map();

module.exports = {
  name: Events.MessageCreate,
  async execute(msg) {
    if (msg.author.bot) return;

    const botId = msg.client.user.id;

    const mentionedBot = msg.mentions.has(botId);
    const repliedToBot = msg.reference?.messageId
      ? (await msg.channel.messages.fetch(msg.reference.messageId)).author.id === botId
      : false;

    if (!mentionedBot && !repliedToBot) return;

    const userPrompt = msg.content.replace(/<@!?(\d+)>/, "").trim();
    const key = `${msg.channel.id}-${msg.author.id}`
    const contextHistory = contextMap.get(key) || [];

    const updatedContext = repliedToBot
      ? [...contextHistory, { prompt: userPrompt }]
      : [{ prompt: userPrompt }];

    const trimmedContext = updatedContext.slice(-MAX_CONTEXT_TURNS);

    try {
      await msg.channel.sendTyping();
      const reply = await getAIResponse(userPrompt, trimmedContext);
      await msg.reply(reply);

      contextMap.set(key, [
        ...trimmedContext.slice(-MAX_CONTEXT_TURNS + 1),
        { prompt: userPrompt, response: reply }
      ]);
    } catch (err) {
      console.error("Error generating response:", err);
      msg.reply("Something went wrong while generating a response.");
    }
  }
};