<script setup>
import { ref } from 'vue'

const question = ref('')
const loading = ref(false)
const error = ref('')
const result = ref(null)

async function diagnose() {
  const q = question.value?.trim()
  if (!q) {
    error.value = '请输入故障现象或问题'
    return
  }
  loading.value = true
  error.value = ''
  result.value = null
  try {
    const res = await fetch('/api/diagnose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || res.statusText || `请求失败 ${res.status}`)
    }
    result.value = await res.json()
  } catch (e) {
    error.value = e.message || '诊断请求失败，请确认后端已启动（run.py --api）'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="app">
    <header class="header">
      <h1>机床故障诊断 AI 助手</h1>
    </header>
    <main class="main">
      <section class="input-section">
        <label for="question">请输入故障现象或问题：</label>
        <textarea
          id="question"
          v-model="question"
          placeholder="例如：主轴异响、加工精度下降..."
          rows="4"
          :disabled="loading"
        />
        <button type="button" class="btn" :disabled="loading" @click="diagnose">
          {{ loading ? '诊断中…' : '开始诊断' }}
        </button>
      </section>
      <p v-if="error" class="error">{{ error }}</p>
      <section v-if="result" class="result-section">
        <h2>诊断意见</h2>
        <div class="answer">{{ result.final_answer }}</div>
        <details v-if="result.retrieved_docs?.length" class="docs">
          <summary>查看检索到的参考资料</summary>
          <ul>
            <li v-for="(doc, i) in result.retrieved_docs" :key="i" class="doc-item">
              {{ doc.content }}
            </li>
          </ul>
        </details>
      </section>
    </main>
  </div>
</template>

<style scoped>
.app {
  max-width: 720px;
  margin: 0 auto;
  padding: 1.5rem;
  font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.header {
  margin-bottom: 1.5rem;
}
.header h1 {
  font-size: 1.5rem;
  font-weight: 600;
  color: #1a1a2e;
}
.input-section label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 500;
  color: #333;
}
textarea {
  width: 100%;
  box-sizing: border-box;
  padding: 0.75rem;
  border: 1px solid #ccc;
  border-radius: 8px;
  font-size: 1rem;
  resize: vertical;
  margin-bottom: 0.75rem;
}
textarea:focus {
  outline: none;
  border-color: #2563eb;
  box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2);
}
.btn {
  padding: 0.6rem 1.25rem;
  background: #2563eb;
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  cursor: pointer;
}
.btn:hover:not(:disabled) {
  background: #1d4ed8;
}
.btn:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}
.error {
  color: #dc2626;
  margin-top: 0.75rem;
}
.result-section {
  margin-top: 1.5rem;
  padding: 1rem;
  background: #f8fafc;
  border-radius: 8px;
}
.result-section h2 {
  font-size: 1.1rem;
  margin-bottom: 0.5rem;
  color: #1e293b;
}
.answer {
  white-space: pre-wrap;
  line-height: 1.6;
  color: #334155;
}
.docs {
  margin-top: 1rem;
}
.docs summary {
  cursor: pointer;
  font-weight: 500;
  color: #475569;
}
.doc-item {
  margin-top: 0.5rem;
  padding: 0.5rem;
  background: #fff;
  border-radius: 4px;
  font-size: 0.9rem;
  color: #64748b;
  list-style: none;
  border-left: 3px solid #e2e8f0;
}
</style>
