/**
 * AI Agent Dashboard - Main JavaScript Application
 */

// Global application state
const AppState = {
    socket: null,
    currentTab: 'overview',
    isConnected: false,
    lastUpdate: null,
    autoRefresh: true,
    refreshInterval: 5000, // 5 seconds
    charts: {},
    intervals: {}
};

// Application initialization
function initializeApp() {
    console.log('Initializing AI Agent Dashboard...');
    
    // Initialize WebSocket connection
    initializeWebSocket();
    
    // Setup tab navigation
    setupTabNavigation();
    
    // Setup form handlers
    setupFormHandlers();
    
    // Setup periodic updates
    setupPeriodicUpdates();
    
    // Load initial data
    loadInitialData();
    
    console.log('AI Agent Dashboard initialized successfully');
}

// WebSocket connection management
function initializeWebSocket() {
    try {
        AppState.socket = io();
        
        AppState.socket.on('connect', function() {
            console.log('Connected to AI Agent');
            AppState.isConnected = true;
            updateConnectionStatus('connected');
            loadDashboardData();
        });
        
        AppState.socket.on('disconnect', function() {
            console.log('Disconnected from AI Agent');
            AppState.isConnected = false;
            updateConnectionStatus('disconnected');
        });
        
        AppState.socket.on('status_update', function(data) {
            updateSystemStatus(data);
        });
        
        AppState.socket.on('command_result', function(data) {
            handleCommandResult(data);
        });
        
        AppState.socket.on('log', function(data) {
            addLogEntry(data);
        });
        
        AppState.socket.on('error', function(error) {
            console.error('WebSocket error:', error);
            showNotification('Connection error: ' + error.message, 'error');
        });
        
    } catch (error) {
        console.error('Failed to initialize WebSocket:', error);
        updateConnectionStatus('disconnected');
    }
}

// Connection status management
function updateConnectionStatus(status) {
    const statusElement = document.getElementById('connectionStatus');
    const statusText = {
        'connected': 'Connected',
        'connecting': 'Connecting...',
        'disconnected': 'Disconnected'
    };
    
    statusElement.textContent = statusText[status] || 'Unknown';
    statusElement.className = `badge bg-${getStatusColor(status)}`;
    statusElement.classList.add(status);
}

function getStatusColor(status) {
    const colors = {
        'connected': 'success',
        'connecting': 'warning',
        'disconnected': 'danger'
    };
    return colors[status] || 'secondary';
}

// Tab navigation
function setupTabNavigation() {
    const tabLinks = document.querySelectorAll('[data-tab]');
    
    tabLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const tabName = this.getAttribute('data-tab');
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    
    // Show selected tab content
    const selectedTab = document.getElementById(tabName);
    if (selectedTab) {
        selectedTab.classList.add('active');
        selectedTab.classList.add('fade-in');
    }
    
    // Add active class to selected nav link
    const selectedLink = document.querySelector(`[data-tab="${tabName}"]`);
    if (selectedLink) {
        selectedLink.classList.add('active');
    }
    
    AppState.currentTab = tabName;
    
    // Load tab-specific data
    loadTabData(tabName);
}

// Form handlers
function setupFormHandlers() {
    // Command form
    const commandForm = document.getElementById('commandForm');
    if (commandForm) {
        commandForm.addEventListener('submit', handleCommandSubmit);
    }
    
    // Model settings form
    const modelSettings = document.getElementById('modelSettings');
    if (modelSettings) {
        modelSettings.addEventListener('submit', handleModelSettingsSubmit);
    }
    
    // System settings form
    const systemSettings = document.getElementById('systemSettings');
    if (systemSettings) {
        systemSettings.addEventListener('submit', handleSystemSettingsSubmit);
    }
}

// Command submission
async function handleCommandSubmit(e) {
    e.preventDefault();
    
    const commandInput = document.getElementById('commandInput');
    const command = commandInput.value.trim();
    
    if (!command) {
        showNotification('Please enter a command', 'warning');
        return;
    }
    
    showLoadingModal();
    
    try {
        const response = await fetch('/api/command', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                command: command,
                priority: document.getElementById('prioritySelect').value,
                category: document.getElementById('categorySelect').value
            })
        });
        
        const result = await response.json();
        
        hideLoadingModal();
        
        if (result.success !== false) {
            showCommandResult(command, result);
            addToCommandHistory(command, result);
            commandInput.value = '';
            showNotification('Command executed successfully', 'success');
        } else {
            showNotification('Command failed: ' + (result.error || 'Unknown error'), 'error');
        }
        
    } catch (error) {
        hideLoadingModal();
        console.error('Command execution error:', error);
        showNotification('Failed to execute command: ' + error.message, 'error');
    }
}

// Quick command execution
function executeQuickCommand(command) {
    document.getElementById('commandInput').value = command;
    document.getElementById('commandForm').dispatchEvent(new Event('submit'));
}

// Command result handling
function handleCommandResult(data) {
    if (data && data.result) {
        showCommandResult(data.command || 'Unknown command', data.result);
    }
}

function showCommandResult(command, result) {
    const resultsContainer = document.getElementById('commandResults');
    const resultElement = createCommandResultElement(command, result);
    
    // Clear placeholder content
    const placeholder = resultsContainer.querySelector('.text-center.text-muted');
    if (placeholder) {
        placeholder.remove();
    }
    
    // Add new result at the top
    resultsContainer.insertBefore(resultElement, resultsContainer.firstChild);
    
    // Limit number of results shown
    const results = resultsContainer.querySelectorAll('.command-result-item');
    if (results.length > 10) {
        results[results.length - 1].remove();
    }
}

function createCommandResultElement(command, result) {
    const element = document.createElement('div');
    element.className = 'command-result-item fade-in';
    
    const isSuccess = result.success !== false && !result.error;
    const statusClass = isSuccess ? 'success' : 'error';
    const statusIcon = isSuccess ? 'check-circle' : 'x-circle';
    
    element.innerHTML = `
        <div class="command-result-header">
            <span class="badge bg-${isSuccess ? 'success' : 'danger'}">
                <i data-feather="${statusIcon}"></i>
                ${isSuccess ? 'Success' : 'Error'}
            </span>
            <small class="text-muted">${new Date().toLocaleTimeString()}</small>
        </div>
        <div class="command-text">${escapeHtml(command)}</div>
        <div class="result-content ${statusClass}">
            ${formatCommandResult(result)}
        </div>
    `;
    
    // Re-initialize feather icons
    feather.replace();
    
    return element;
}

function formatCommandResult(result) {
    if (result.error) {
        return `<strong>Error:</strong> ${escapeHtml(result.error)}`;
    }
    
    if (result.result && typeof result.result === 'object') {
        return `<pre>${JSON.stringify(result.result, null, 2)}</pre>`;
    }
    
    if (result.result) {
        return escapeHtml(result.result.toString());
    }
    
    return 'Command completed successfully';
}

// Command history management
function addToCommandHistory(command, result) {
    const historyContainer = document.getElementById('commandHistory');
    const historyElement = createHistoryElement(command, result);
    
    // Clear placeholder content
    const placeholder = historyContainer.querySelector('.text-center.text-muted');
    if (placeholder) {
        placeholder.remove();
    }
    
    // Add new history item at the top
    historyContainer.insertBefore(historyElement, historyContainer.firstChild);
    
    // Limit history items
    const historyItems = historyContainer.querySelectorAll('.history-item');
    if (historyItems.length > 20) {
        historyItems[historyItems.length - 1].remove();
    }
}

function createHistoryElement(command, result) {
    const element = document.createElement('div');
    element.className = 'history-item cursor-pointer';
    element.onclick = () => {
        document.getElementById('commandInput').value = command;
    };
    
    element.innerHTML = `
        <div class="history-command">${escapeHtml(command)}</div>
        <div class="history-time">${new Date().toLocaleTimeString()}</div>
    `;
    
    return element;
}

// Data loading functions
async function loadInitialData() {
    try {
        await Promise.all([
            loadSystemStatus(),
            loadAvailableModels(),
            loadRecentActivity()
        ]);
    } catch (error) {
        console.error('Failed to load initial data:', error);
    }
}

async function loadDashboardData() {
    if (!AppState.isConnected) return;
    
    try {
        await Promise.all([
            loadSystemStatus(),
            loadTaskStatistics(),
            loadLearningStats(),
            loadSystemHealth()
        ]);
    } catch (error) {
        console.error('Failed to load dashboard data:', error);
    }
}

async function loadTabData(tabName) {
    switch (tabName) {
        case 'overview':
            await loadOverviewData();
            break;
        case 'commands':
            await loadCommandsData();
            break;
        case 'tasks':
            await loadTasksData();
            break;
        case 'learning':
            await loadLearningData();
            break;
        case 'system':
            await loadSystemData();
            break;
        case 'settings':
            await loadSettingsData();
            break;
    }
}

// Specific data loading functions
async function loadSystemStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();
        
        updateSystemStatusDisplay(status);
    } catch (error) {
        console.error('Failed to load system status:', error);
    }
}

function updateSystemStatusDisplay(status) {
    // Update status indicators
    const agentStatus = document.getElementById('agentStatus');
    if (agentStatus) {
        agentStatus.textContent = status.active ? 'Active' : 'Inactive';
        agentStatus.className = status.active ? 'text-success' : 'text-danger';
    }
    
    const currentModel = document.getElementById('currentModel');
    if (currentModel) {
        currentModel.textContent = status.current_model || 'Unknown';
    }
    
    const activeTasks = document.getElementById('activeTasks');
    if (activeTasks) {
        activeTasks.textContent = status.tasks_running || 0;
    }
    
    const memoryUsage = document.getElementById('memoryUsage');
    if (memoryUsage && status.memory_usage) {
        memoryUsage.textContent = `${Math.round(status.memory_usage.rss || 0)} MB`;
    }
}

async function loadAvailableModels() {
    try {
        const response = await fetch('/api/models');
        const models = await response.json();
        
        updateModelSelect(models);
    } catch (error) {
        console.error('Failed to load available models:', error);
    }
}

function updateModelSelect(models) {
    const modelSelect = document.getElementById('modelSelect');
    if (!modelSelect) return;
    
    // Clear existing options
    modelSelect.innerHTML = '';
    
    // Add model options
    Object.entries(models).forEach(([name, info]) => {
        if (info.available) {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = `${name} (${info.type})`;
            modelSelect.appendChild(option);
        }
    });
}

async function loadRecentActivity() {
    try {
        // This would fetch recent activity from the API
        const activities = [
            {
                icon: 'terminal',
                title: 'Command Executed',
                description: 'System status command completed',
                time: '2 minutes ago',
                type: 'success'
            },
            {
                icon: 'brain',
                title: 'Learning Update',
                description: 'Pattern analysis completed',
                time: '5 minutes ago',
                type: 'info'
            },
            {
                icon: 'clock',
                title: 'Task Scheduled',
                description: 'Backup task scheduled for tonight',
                time: '10 minutes ago',
                type: 'warning'
            }
        ];
        
        displayRecentActivity(activities);
    } catch (error) {
        console.error('Failed to load recent activity:', error);
    }
}

function displayRecentActivity(activities) {
    const container = document.getElementById('recentActivity');
    if (!container) return;
    
    container.innerHTML = '';
    
    activities.forEach(activity => {
        const element = createActivityElement(activity);
        container.appendChild(element);
    });
}

function createActivityElement(activity) {
    const element = document.createElement('div');
    element.className = 'activity-item';
    
    const iconColors = {
        'success': 'bg-success',
        'info': 'bg-info',
        'warning': 'bg-warning',
        'error': 'bg-danger'
    };
    
    element.innerHTML = `
        <div class="activity-icon ${iconColors[activity.type]} text-white">
            <i data-feather="${activity.icon}"></i>
        </div>
        <div class="activity-content">
            <div class="activity-title">${activity.title}</div>
            <div class="activity-description">${activity.description}</div>
            <div class="activity-time">${activity.time}</div>
        </div>
    `;
    
    feather.replace();
    return element;
}

async function loadSystemHealth() {
    try {
        // Simulate system health data
        const health = {
            cpu: Math.random() * 100,
            memory: Math.random() * 100,
            disk: Math.random() * 100
        };
        
        updateSystemHealth(health);
    } catch (error) {
        console.error('Failed to load system health:', error);
    }
}

function updateSystemHealth(health) {
    updateProgressBar('cpuProgress', health.cpu);
    updateProgressBar('memoryProgress', health.memory);
    updateProgressBar('diskProgress', health.disk);
}

function updateProgressBar(elementId, value) {
    const progressBar = document.getElementById(elementId);
    if (!progressBar) return;
    
    progressBar.style.width = `${value}%`;
    progressBar.setAttribute('aria-valuenow', value);
    
    // Update color based on value
    progressBar.className = 'progress-bar';
    if (value > 90) {
        progressBar.classList.add('bg-danger');
    } else if (value > 75) {
        progressBar.classList.add('bg-warning');
    } else {
        progressBar.classList.add('bg-success');
    }
}

// Modal management
function showLoadingModal() {
    const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
    modal.show();
}

function hideLoadingModal() {
    const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
    if (modal) {
        modal.hide();
    }
}

// Notification system
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

// Periodic updates
function setupPeriodicUpdates() {
    if (AppState.autoRefresh) {
        AppState.intervals.statusUpdate = setInterval(() => {
            if (AppState.isConnected) {
                loadSystemStatus();
                loadSystemHealth();
            }
        }, AppState.refreshInterval);
    }
}

// Quick action functions
function showCommandInterface() {
    switchTab('commands');
    document.getElementById('commandInput').focus();
}

function scheduleTask() {
    switchTab('tasks');
    showNotification('Task scheduling interface opened', 'info');
}

function generateContent() {
    const command = 'generate a blog post about AI automation';
    document.getElementById('commandInput').value = command;
    switchTab('commands');
}

function systemScan() {
    const command = 'perform system health check and security scan';
    executeQuickCommand(command);
}

function exportData() {
    showNotification('Data export initiated', 'info');
    // Implementation would download export file
}

function clearCommand() {
    document.getElementById('commandInput').value = '';
}

// Settings management
async function handleModelSettingsSubmit(e) {
    e.preventDefault();
    
    const model = document.getElementById('modelSelect').value;
    const temperature = document.getElementById('temperatureSlider').value;
    
    try {
        const response = await fetch(`/api/models/${model}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                temperature: parseFloat(temperature)
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            showNotification('Model settings updated successfully', 'success');
        } else {
            showNotification('Failed to update model settings', 'error');
        }
    } catch (error) {
        console.error('Settings update error:', error);
        showNotification('Error updating settings: ' + error.message, 'error');
    }
}

async function handleSystemSettingsSubmit(e) {
    e.preventDefault();
    
    const settings = {
        auto_update: document.getElementById('autoUpdateCheck').checked,
        voice_enabled: document.getElementById('voiceEnabledCheck').checked,
        learning_enabled: document.getElementById('learningEnabledCheck').checked,
        log_level: document.getElementById('logLevelSelect').value
    };
    
    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        });
        
        const result = await response.json();
        
        if (result.success) {
            showNotification('System settings updated successfully', 'success');
        } else {
            showNotification('Failed to update system settings', 'error');
        }
    } catch (error) {
        console.error('Settings update error:', error);
        showNotification('Error updating settings: ' + error.message, 'error');
    }
}

// Utility functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
}

// Log entry handling
function addLogEntry(logData) {
    const logsContainer = document.getElementById('systemLogs');
    if (!logsContainer) return;
    
    // Clear placeholder if exists
    const placeholder = logsContainer.querySelector('.text-center.text-muted');
    if (placeholder) {
        placeholder.remove();
    }
    
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${logData.level || 'info'}`;
    
    const timestamp = new Date(logData.timestamp || Date.now()).toLocaleTimeString();
    
    logEntry.innerHTML = `
        <span class="log-timestamp">${timestamp}</span>
        <span class="log-level">${(logData.level || 'INFO').toUpperCase()}</span>
        ${escapeHtml(logData.entry || logData.message || '')}
    `;
    
    logsContainer.appendChild(logEntry);
    
    // Limit log entries
    const entries = logsContainer.querySelectorAll('.log-entry');
    if (entries.length > 100) {
        entries[0].remove();
    }
    
    // Auto-scroll to bottom
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

// Additional tab-specific data loading functions
async function loadOverviewData() {
    await Promise.all([
        loadSystemStatus(),
        loadRecentActivity(),
        loadSystemHealth()
    ]);
}

async function loadCommandsData() {
    // Load command history and available quick commands
}

async function loadTasksData() {
    try {
        const response = await fetch('/api/tasks');
        const tasks = await response.json();
        
        updateTasksDisplay(tasks);
    } catch (error) {
        console.error('Failed to load tasks data:', error);
    }
}

async function loadLearningData() {
    try {
        const [statsResponse, memoryResponse] = await Promise.all([
            fetch('/api/learning/stats'),
            fetch('/api/memory/summary')
        ]);
        
        const stats = await statsResponse.json();
        const memory = await memoryResponse.json();
        
        updateLearningDisplay(stats, memory);
    } catch (error) {
        console.error('Failed to load learning data:', error);
    }
}

async function loadSystemData() {
    // Load system monitoring data, logs, and performance metrics
}

async function loadSettingsData() {
    try {
        const response = await fetch('/api/settings');
        const settings = await response.json();
        
        updateSettingsDisplay(settings);
    } catch (error) {
        console.error('Failed to load settings data:', error);
    }
}

function updateTasksDisplay(tasks) {
    // Implementation for updating tasks display
}

function updateLearningDisplay(stats, memory) {
    const totalInteractions = document.getElementById('totalInteractions');
    if (totalInteractions && stats.total_interactions !== undefined) {
        totalInteractions.textContent = stats.total_interactions;
    }
    
    const patternsDiscovered = document.getElementById('patternsDiscovered');
    if (patternsDiscovered && stats.patterns_discovered !== undefined) {
        patternsDiscovered.textContent = stats.patterns_discovered;
    }
}

function updateSettingsDisplay(settings) {
    // Update settings form with current values
}

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    // Clear intervals
    Object.values(AppState.intervals).forEach(interval => {
        clearInterval(interval);
    });
    
    // Close WebSocket connection
    if (AppState.socket) {
        AppState.socket.disconnect();
    }
});

// Export global functions for HTML onclick handlers
window.showCommandInterface = showCommandInterface;
window.scheduleTask = scheduleTask;
window.generateContent = generateContent;
window.systemScan = systemScan;
window.exportData = exportData;
window.clearCommand = clearCommand;
window.executeQuickCommand = executeQuickCommand;
