#!/usr/bin/env node
// Computes the TensorFold pack source identity (sha256 of the identity string
// from zig/src/families/flashnext/pack_io.zig) for a sharded safetensors model.
//
// Usage: node compute_pack_identity.js <indexSha256> <configSha256> <shard1Name> <shard1Size> <shard1HeaderSha> [<shard2Name> <shard2Size> <shard2HeaderSha> ...]
const crypto = require('crypto');
const sha = s => crypto.createHash('sha256').update(s).digest('hex');

const args = process.argv.slice(2);
const idx = args[0];
const cfg = args[1];
const shards = [];
for (let i = 2; i < args.length; i += 3) {
  shards.push([args[i], BigInt(args[i + 1]).toString(), args[i + 2]]);
}
const sorted = shards.slice().sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
let s = 'flashnext pack 3\n';
s += 'index ' + idx + '\n';
s += 'config ' + cfg + '\n';
for (const [n, sz, h] of sorted) s += 'shard ' + n + ' ' + sz + ' ' + h + '\n';
console.log(s);
console.log('identitySha256 =', sha(s));
