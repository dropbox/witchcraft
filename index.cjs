class NoopWarp {
  constructor(_dbname) {
    console.warn("*** constructing no-op Warp instance ***");
  }
  async search(query) {
    return [];
  }
}

try {
  module.exports = require('./warp.node');
  console.log("Warp module successfully loaded", module.exports);
} catch {
  console.log("failed to load warp.node");
  module.exports.Warp = NoopWarp;
}
