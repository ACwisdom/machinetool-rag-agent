# 机床故障诊断 AI 助手 - 前端

Vue 3 + Vite，对接后端 FastAPI `/diagnose` 接口。

## 开发

1. **先启动后端**（项目根目录）：
   ```bash
   python run.py --api
   ```
   服务地址：http://127.0.0.1:8000

2. **再启动前端**：
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   页面地址：http://localhost:5173  
   前端会通过 Vite 代理将 `/api/*` 转发到后端，无需配置 CORS。

## 构建

```bash
npm run build
```
产物在 `dist/`，可交给任意静态服务器或与后端同域部署。

## 生产环境 API 地址

开发时使用 Vite 代理；若生产环境前后端分离部署，需在构建前配置 API 基础地址（例如环境变量 `VITE_API_BASE` + 在代码里用 `import.meta.env.VITE_API_BASE`）。
