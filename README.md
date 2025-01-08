
### 数据集


### 运行程序

需要在 python3.7  + GPU （安装好 nvidia driver、cuda、cudnn） 环境中

本实验运行时的依赖版本为 cuda 10.0, cudnn 7.6.0

```bash
# 创建环境
conda create -n tf1 python=3.7
conda activate tf1
# 安装 tensorflow-gpu
pip install tensorflow-gpu==1.15.5
# 配置 cuda && cudnn (因为安装了高版本的驱动，所以通过 conda 配置即可成功)
conda install cudatoolkit=10.0
conda install cudnn=7.6.0
# 安装 pytorch
conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
# 安装其他依赖
pip install -r requirements.txt
```

### 数据库配置

在运行防御或攻击方法时会自动添加一条记录到 mysql 数据库中，所以在运行前请修改 `config.ini` 中的配置

### Web UI

有一个简单的 UI 界面用来展示攻击/防御之后的结果和运行耗时情况，具体脚本在 `ui/web_ui.sh`，运行或者停止都要在项目根目录下执行命令：

```bash
cd {projectRoot}  # projectRoot 替换成项目的根目录
bash ui/web_ui.sh start   # 启动
bash ui/web_ui.sh stop    # 停止
bash ui/web_ui.sh restart # 重启
```


