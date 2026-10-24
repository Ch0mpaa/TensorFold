package expo.modules.tensorfoldlocalruntime;

import android.os.Bundle;

interface IQnnExecutor {
  Bundle createGraph();
  Bundle execute(long graphHandle, int repeats);
  void releaseGraph(long graphHandle);
  Bundle health();
}
