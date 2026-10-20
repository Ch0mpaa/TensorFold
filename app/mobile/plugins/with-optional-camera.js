const { withAndroidManifest } = require('expo/config-plugins')

module.exports = function withOptionalCamera(config) {
  return withAndroidManifest(config, (next) => {
    const manifest = next.modResults.manifest
    const features = manifest['uses-feature'] || []
    if (!features.some((feature) => feature.$?.['android:name'] === 'android.hardware.camera')) {
      features.push({ $: { 'android:name': 'android.hardware.camera', 'android:required': 'false' } })
    }
    manifest['uses-feature'] = features
    return next
  })
}
