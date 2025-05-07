### 0MP 错误
- 详情：OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
- 解决：
  - import os
  - os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'