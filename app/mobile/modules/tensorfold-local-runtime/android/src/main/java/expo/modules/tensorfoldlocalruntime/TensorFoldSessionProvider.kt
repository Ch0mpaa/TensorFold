package expo.modules.tensorfoldlocalruntime

import android.content.ContentProvider
import android.content.ContentValues
import android.database.Cursor
import android.database.MatrixCursor
import android.net.Uri

/** Explicit-permission handoff for the separately installed PwnHarness app. */
class TensorFoldSessionProvider : ContentProvider() {
  override fun onCreate(): Boolean = true

  override fun query(
    uri: Uri,
    projection: Array<out String>?,
    selection: String?,
    selectionArgs: Array<out String>?,
    sortOrder: String?,
  ): Cursor {
    if (uri.path != "/session") throw IllegalArgumentException("Unknown TensorFold provider path.")
    val columns = arrayOf("endpoint", "bearer_token", "model_id", "state")
    val cursor = MatrixCursor(columns, 1)
    RuntimeManager.harnessSession()?.let { session ->
      cursor.addRow(columns.map { session[it] })
    }
    return cursor
  }

  override fun getType(uri: Uri): String = "vnd.android.cursor.item/vnd.tensorfold.session"
  override fun insert(uri: Uri, values: ContentValues?): Uri? = throw UnsupportedOperationException("Read only")
  override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?): Int = throw UnsupportedOperationException("Read only")
  override fun update(uri: Uri, values: ContentValues?, selection: String?, selectionArgs: Array<out String>?): Int = throw UnsupportedOperationException("Read only")
}
