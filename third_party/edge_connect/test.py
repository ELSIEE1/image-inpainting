import os
from src.config import Config
from src.edge_connect import EdgeConnect

def main():
    # 载入配置文件
    config_path = './checkpoints/config.yml'
    config = Config(config_path)
    print("=== DEBUG CONFIG ===")
    print("TEST_FLIST:", config.TEST_FLIST)
    print("TEST_EDGE_FLIST:", config.TEST_EDGE_FLIST)
    print("TEST_MASK_FLIST:", config.TEST_MASK_FLIST)
    print("RESULTS:", config.RESULTS)
    print("====================")

    # 强制使用测试模式
    config.MODE = 2

    # 创建模型
    model = EdgeConnect(config)

    print("Loading checkpoint...")
    model.load()

    print("\nstart testing...\n")

    # 运行测试（官方接口）
    model.test()


if __name__ == "__main__":
    main()
