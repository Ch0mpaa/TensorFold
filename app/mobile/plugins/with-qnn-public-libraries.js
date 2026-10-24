const { withAndroidManifest } = require('expo/config-plugins')

module.exports = function withQnnPublicLibraries(config) {
  return withAndroidManifest(config, (next) => {
    const app = next.modResults.manifest.application?.[0]
    if (!app) throw new Error('Android manifest has no application element')
    const libraries = app['uses-native-library'] || []
    for (const name of ['libcdsprpc.so', 'libadsprpc.so']) {
      if (!libraries.some((library) => library.$?.['android:name'] === name)) {
        libraries.push({ $: { 'android:name': name, 'android:required': 'false' } })
      }
    }
    app['uses-native-library'] = libraries
    return next
  })
}
