const { Events } = require('discord.js');
const { getAIResponse } = require("../utils/ai")

module.exports = {
  name: Events.MessageCreate,
  async execute(msg) {
    if (msg.author.bot) return;
    if (!msg.content.startsWith("!")) return;

    const prompt = msg.content.slice(1).trim();

    try {
      await msg.channel.sendTyping();
      const reply = await getAIResponse(prompt);
      msg.reply(reply);
    } catch (err) {
      console.error("Error generating response:", err);
      msg.reply("Something went wrong while generating a response.");
    }
  }
};