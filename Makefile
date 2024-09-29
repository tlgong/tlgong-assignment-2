# Makefile

# 定义虚拟环境目录
VENV_DIR = venv

# 定义虚拟环境中的 Python 和 pip 路径
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

# 默认目标
.PHONY: install run

# install 目标：创建虚拟环境并安装依赖
install: $(VENV_DIR)/bin/python
	$(PYTHON) -m install numpy matplotlib flask
	@echo "依赖包安装完成。"

# 规则：如果虚拟环境的 Python 不存在，则创建虚拟环境
$(VENV_DIR)/bin/python:
	@echo "创建虚拟环境..."
	python3 -m venv $(VENV_DIR)
	@echo "虚拟环境创建完成。"
lint:
	@echo "运行 flake8 进行代码检查..."
	$(PYTHON) -m pip install flake8
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=$(VENV_DIR)/*
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude=$(VENV_DIR)/*
	@echo "代码检查完成
# run 目标：运行 Flask 应用
run:
	@echo "启动 Flask 应用..."
	$(PYTHON) app.py
