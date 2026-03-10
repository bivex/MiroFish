<template>
  <div class="main-view">
    <!-- Header -->
    <header class="app-header">
      <div class="header-left">
        <div class="brand" @click="router.push('/')">MIROFISH</div>
      </div>
      
      <div class="header-center">
        <div class="view-switcher">
          <button 
            v-for="mode in ['graph', 'split', 'workbench']" 
            :key="mode"
            class="switch-btn"
            :class="{ active: viewMode === mode }"
            @click="viewMode = mode"
          >
            {{ { graph: 'Graph', split: 'Split', workbench: 'Workbench' }[mode] }}
          </button>
        </div>
      </div>

      <div class="header-right">
        <div class="workflow-step">
          <span class="step-num">Step 5/5</span>
          <span class="step-name">Deep Interaction</span>
        </div>
        <div class="step-divider"></div>
        <span class="status-indicator" :class="statusClass">
          <span class="dot"></span>
          {{ statusText }}
        </span>
      </div>
    </header>

    <!-- Main Content Area -->
    <main class="content-area">
      <!-- Left Panel: Graph -->
      <div v-if="shouldRenderGraphPanel" class="panel-wrapper left" :style="leftPanelStyle">
        <GraphPanel 
          :graphData="graphData"
          :loading="graphLoading"
          :currentPhase="5"
          :isSimulating="false"
          :emptyStateMessage="graphEmptyStateMessage"
          :emptyStateHint="graphEmptyStateHint"
          @refresh="refreshGraph"
          @toggle-maximize="toggleMaximize('graph')"
        />
      </div>

      <!-- Right Panel: Step5 深度互动 -->
      <div class="panel-wrapper right" :style="rightPanelStyle">
        <Step5Interaction
          :reportId="currentReportId"
          :simulationId="simulationId"
          :systemLogs="systemLogs"
          @add-log="addLog"
          @update-status="updateStatus"
        />
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, computed, watch, defineAsyncComponent } from 'vue'
import { useRoute, useRouter } from 'vue-router'
const GraphPanel = defineAsyncComponent(() => import('../components/GraphPanel.vue'))
import Step5Interaction from '../components/Step5Interaction.vue'
import { getProject, getGraphData } from '../api/graph'
import { getSimulation } from '../api/simulation'
import { getReport } from '../api/report'

const route = useRoute()
const router = useRouter()

// Props
const props = defineProps({
  reportId: String
})

// Layout State - 默认切换到工作台视角
const viewMode = ref('workbench')

// Data State
const currentReportId = ref(route.params.reportId)
const simulationId = ref(null)
const projectData = ref(null)
const graphData = ref(null)
const graphLoading = ref(false)
const graphContextLoading = ref(false)
const graphContextLoaded = ref(false)
const graphLoadError = ref('')
const systemLogs = ref([])
const currentStatus = ref('ready') // ready | processing | completed | error

// --- Computed Layout Styles ---
const leftPanelStyle = computed(() => {
  if (viewMode.value === 'graph') return { width: '100%', opacity: 1, transform: 'translateX(0)' }
  if (viewMode.value === 'workbench') return { width: '0%', opacity: 0, transform: 'translateX(-20px)' }
  return { width: '50%', opacity: 1, transform: 'translateX(0)' }
})

const rightPanelStyle = computed(() => {
  if (viewMode.value === 'workbench') return { width: '100%', opacity: 1, transform: 'translateX(0)' }
  if (viewMode.value === 'graph') return { width: '0%', opacity: 0, transform: 'translateX(20px)' }
  return { width: '50%', opacity: 1, transform: 'translateX(0)' }
})

const shouldRenderGraphPanel = computed(() => viewMode.value !== 'workbench')

// --- Status Computed ---
const statusClass = computed(() => {
  return currentStatus.value
})

const statusText = computed(() => {
  if (currentStatus.value === 'error') return 'Error'
  if (currentStatus.value === 'completed') return 'Completed'
  if (currentStatus.value === 'processing') return 'Processing'
  return 'Ready'
})

const graphEmptyStateMessage = computed(() => {
  if (graphLoadError.value) return 'Unable to load graph data.'
  if (Array.isArray(graphData.value?.nodes) && graphData.value.nodes.length === 0) {
    return 'The graph loaded, but there are no renderable nodes yet.'
  }
  if (projectData.value?.graph_id) return 'Graph data is temporarily unavailable in this view.'
  if (projectData.value?.ontology) return 'Ontology is ready, but the graph build has not completed yet.'
  if (projectData.value) return 'Graph data has not been generated for this project yet.'
  return 'Loading interaction context...'
})

const graphEmptyStateHint = computed(() => {
  if (graphLoadError.value) return graphLoadError.value
  if (Array.isArray(graphData.value?.nodes) && graphData.value.nodes.length === 0) {
    return 'Try refreshing after more entities have been ingested into the graph.'
  }
  if (projectData.value?.graph_id) return 'This project already has a graph id. Use Refresh to retry loading it.'
  if (projectData.value?.ontology) return 'Refresh after the graph build finishes.'
  return ''
})

// --- Helpers ---
const addLog = (msg) => {
  const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }) + '.' + new Date().getMilliseconds().toString().padStart(3, '0')
  systemLogs.value.push({ time, msg })
  if (systemLogs.value.length > 200) {
    systemLogs.value.shift()
  }
}

const updateStatus = (status) => {
  currentStatus.value = status
}

// --- Layout Methods ---
const toggleMaximize = (target) => {
  if (viewMode.value === target) {
    viewMode.value = 'split'
  } else {
    viewMode.value = target
  }
}

// --- Data Logic ---
const loadReportData = async () => {
  try {
    graphLoadError.value = ''
    addLog(`Loading report data: ${currentReportId.value}`)
    
    // 获取 report 信息以获取 simulation_id
    const reportRes = await getReport(currentReportId.value)
    if (reportRes.success && reportRes.data) {
      const reportData = reportRes.data
      simulationId.value = reportData.simulation_id

      if (shouldRenderGraphPanel.value) {
        void ensureGraphContextLoaded()
      }
    } else {
      addLog(`Failed to fetch report info: ${reportRes.error || 'Unknown error'}`)
    }
  } catch (err) {
    addLog(`Load error: ${err.message}`)
  }
}

const ensureGraphContextLoaded = async () => {
  if (!simulationId.value || graphContextLoaded.value || graphContextLoading.value) return

  graphContextLoading.value = true
  graphLoadError.value = ''

  try {
    const simRes = await getSimulation(simulationId.value)
    if (!(simRes.success && simRes.data)) {
      throw new Error(simRes.error || 'Failed to load simulation details')
    }

    const simData = simRes.data
    if (!simData.project_id) {
      graphContextLoaded.value = true
      return
    }

    const projRes = await getProject(simData.project_id)
    if (!(projRes.success && projRes.data)) {
      throw new Error(projRes.error || 'Failed to load project details')
    }

    projectData.value = projRes.data
    graphContextLoaded.value = true
    addLog(`Project loaded: ${projRes.data.project_id}`)

    if (projRes.data.graph_id) {
      await loadGraph(projRes.data.graph_id)
    }
  } catch (err) {
    graphLoadError.value = err.message
    addLog(`Failed to load graph context: ${err.message}`)
  } finally {
    graphContextLoading.value = false
  }
}

const loadGraph = async (graphId) => {
  graphLoading.value = true
  graphLoadError.value = ''
  
  try {
    const res = await getGraphData(graphId)
    if (res.success && res.data) {
      graphData.value = res.data
      addLog('Graph data loaded successfully')
    } else {
      graphData.value = null
      graphLoadError.value = 'The graph endpoint returned an empty payload.'
      addLog('Graph endpoint returned no data')
    }
  } catch (err) {
    graphData.value = null
    graphLoadError.value = err.message
    addLog(`Failed to load graph data: ${err.message}`)
  } finally {
    graphLoading.value = false
  }
}

const refreshGraph = async () => {
  await ensureGraphContextLoaded()

  if (projectData.value?.graph_id) {
    loadGraph(projectData.value.graph_id)
  }
}

watch(shouldRenderGraphPanel, (shouldRender) => {
  if (shouldRender) {
    void ensureGraphContextLoaded()
  }
})

// Watch route params
watch(() => route.params.reportId, (newId) => {
  if (newId && newId !== currentReportId.value) {
    currentReportId.value = newId
    projectData.value = null
    graphData.value = null
    graphLoading.value = false
    graphContextLoading.value = false
    graphContextLoaded.value = false
    graphLoadError.value = ''
    loadReportData()
  }
}, { immediate: true })
</script>

<style scoped>
.main-view {
  height: 100vh;
  height: 100dvh;
  display: flex;
  flex-direction: column;
  background: #FFF;
  overflow: hidden;
  font-family: 'Space Grotesk', 'Noto Sans SC', system-ui, sans-serif;
}

/* Header */
.app-header {
  height: 60px;
  border-bottom: 1px solid #EAEAEA;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  background: #FFF;
  z-index: 100;
  position: relative;
}

.header-center {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
}

.brand {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 800;
  font-size: 18px;
  letter-spacing: 1px;
  cursor: pointer;
}

.view-switcher {
  display: flex;
  background: #F5F5F5;
  padding: 4px;
  border-radius: 6px;
  gap: 4px;
}

.switch-btn {
  border: none;
  background: transparent;
  padding: 6px 16px;
  font-size: 12px;
  font-weight: 600;
  color: #666;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
}

.switch-btn.active {
  background: #FFF;
  color: #000;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.workflow-step {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.step-num {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
  color: #999;
}

.step-name {
  font-weight: 700;
  color: #000;
}

.step-divider {
  width: 1px;
  height: 14px;
  background-color: #E0E0E0;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #666;
  font-weight: 500;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #CCC;
}

.status-indicator.ready .dot { background: #4CAF50; }
.status-indicator.processing .dot { background: #FF9800; animation: pulse 1s infinite; }
.status-indicator.completed .dot { background: #4CAF50; }
.status-indicator.error .dot { background: #F44336; }

@keyframes pulse { 50% { opacity: 0.5; } }

/* Content */
.content-area {
  flex: 1;
  display: flex;
  position: relative;
  overflow: hidden;
}

.panel-wrapper {
  height: 100%;
  overflow: hidden;
  transition: width 0.4s cubic-bezier(0.25, 0.8, 0.25, 1), opacity 0.3s ease, transform 0.3s ease;
  will-change: width, opacity, transform;
}

.panel-wrapper.left {
  border-right: 1px solid #EAEAEA;
}

@media (max-width: 1024px) {
  .main-view {
    height: auto;
    min-height: 100dvh;
    overflow: auto;
  }

  .app-header {
    height: auto;
    min-height: 60px;
    padding: 14px 16px;
    flex-wrap: wrap;
    gap: 12px;
  }

  .header-center {
    display: none;
  }

  .header-right {
    margin-left: auto;
    gap: 12px;
  }

  .content-area {
    flex-direction: column;
    overflow: visible;
  }

  .panel-wrapper {
    width: 100% !important;
    height: auto;
    min-height: 45vh;
    opacity: 1 !important;
    transform: none !important;
  }

  .panel-wrapper.left {
    border-right: none;
    border-bottom: 1px solid #EAEAEA;
  }
}

@media (max-width: 640px) {
  .app-header {
    align-items: flex-start;
  }

  .header-right {
    width: 100%;
    justify-content: space-between;
    flex-wrap: wrap;
  }

  .workflow-step {
    flex-direction: column;
    align-items: flex-start;
    gap: 2px;
    font-size: 12px;
  }

  .step-divider {
    display: none;
  }

  .status-indicator {
    font-size: 11px;
  }

  .panel-wrapper {
    min-height: 50vh;
  }
}
</style>
