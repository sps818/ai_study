"""
    FASTAPI写法参考
"""
from typing import Union
from fastapi import FastAPI
import joblib
from pydantic import BaseModel


# 加载状态字典
idx2label, gender_dict, education_dict, marital_dict, income_dict, card_dict, mu, sigma = joblib.load(
    filename="../state.lxh")

# 加载模型
dtc_model = joblib.load(filename="../dtc.lxh")
# 创建一个服务
app = FastAPI()

# 自定义一个参数接受类
class Param(BaseModel):
    x: str


class ResultEntity(BaseModel):
    result: str


# 创建一个路由
@app.post("/predict")
def read_item(param: Param):
    x = param.x

    import numpy as np
    # print(f"原始输入：{x}")
    x = x.split(",")[1:]
    # print(f"切分之后：{x}")
    temp = []
    # 1, age
    temp.append(float(x[0]))
    # 2, gender
    temp.append(gender_dict[x[1]])
    # 3,
    temp.append(float(x[2]))
    # 4,
    temp.append(education_dict[x[3]])

    # 5,
    temp.append(marital_dict[x[4]])
    # 6，
    temp.append(income_dict[x[5]])

    # 7,
    temp.append(card_dict[x[6]])

    # 8 - 19
    temp.extend([float(ele) for ele in x[7:]])

    # print(f"编码之后：{temp}")

    # 标准化之后
    x = np.array(temp)
    x = (x - mu) / sigma
    # print(f"标准化之后：{x}")

    # 模型推理

    y_pred = dtc_model.predict(X=[x])
    # print(f"推理结果：{y_pred}")

    # 结果解析
    final_result = idx2label[y_pred[0]]
    # print(f"最终结果：{final_result}")

    return ResultEntity(result=final_result)


# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)