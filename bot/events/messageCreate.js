const { Events } = require('discord.js');

module.exports = {
  name: Events.MessageCreate,
  async execute(msg) {
    if (msg.author.bot) return;
  }
};