require("dotenv").config({ path: "../.env"});
const fs = require("node:fs");
const path = require("node:path")
const { Client, GatewayIntentBits, Collection } = require("discord.js");

const client = new Client({ intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildMessages, GatewayIntentBits.MessageContent ]});

// load commands
client.commands = new Collection();
const commandsPath = path.join(__dirname, "commands");
const commandFiles = fs.readdirSync(commandsPath);
for (const file of commandFiles) {
  const command = require(`./commands/${file}`)
  if ("data" in command && "execute" in command) {
    client.commands.set(command.data.name, command);
  } else {
    console.log(`[WARNING] The command in ${file} is missing required "data" or "execute" property.`)
  }
}

// load events
const eventsPath = path.join(__dirname, "events");
const eventFiles = fs.readdirSync(eventsPath);
for (const file of eventFiles) {
  const event = require(`./events/${file}`)
  if (event.once) {
    client.once(event.name, (...args) => event.execute(...args, client));
  } else {
    client.on(event.name, (...args) => event.execute(...args, client));
  }
}

client.login(process.env.DISCORD_TOKEN);