const warpNode = require('../target/release/warp.node');

warpNode.setLogCallback((r) => {
  console.log(`[warp ${r.file}:${r.line}] ${r.message}`);
}, 'info');
module.warp = new warpNode.Warp("mydb.sqlite", "assets");
console.log("warp", module.warp);


export async function search(query, threshold, top_k, filter) {
    return module.warp.search(query, threshold, top_k, filter);
}
