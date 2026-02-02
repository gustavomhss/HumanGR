/**
 * Cockpit Dashboard - Real-time WebSocket client
 *
 * Connects to Flask-SocketIO server and updates node states in real-time.
 * This is the client-side JavaScript for the cockpit dashboard.
 */
(function() {
    'use strict';

    // ==========================================================================
    // CONFIGURATION
    // ==========================================================================

    const CONFIG = {
        reconnectAttempts: 10,
        reconnectDelay: 2000,
        pollInterval: 5000,  // Fallback REST polling interval
    };

    // ==========================================================================
    // STATE
    // ==========================================================================

    let socket = null;
    let connected = false;
    let reconnectAttempts = 0;

    // Node data store (will be populated from server)
    window.nodeData = window.nodeData || {};

    // Track which events we've already shown (for deduplication)
    let shownEventIds = new Set();
    let shownRunLogLines = new Set();
    let shownRedisEventIds = new Set();

    // ==========================================================================
    // CONNECTION STATUS
    // ==========================================================================

    function updateConnectionStatus(isConnected, message) {
        const statusEl = document.getElementById('connection-status');
        const headerStatusEl = document.getElementById('header-status');

        if (isConnected) {
            if (statusEl) {
                statusEl.className = 'connection-status connected';
                statusEl.title = 'Live WebSocket connection';
            }
            if (headerStatusEl) {
                headerStatusEl.textContent = 'Running';
            }
        } else {
            if (statusEl) {
                statusEl.className = 'connection-status disconnected';
                statusEl.title = message || 'Disconnected';
            }
            if (headerStatusEl) {
                headerStatusEl.textContent = message || 'Connecting...';
            }
        }
    }

    // ==========================================================================
    // SOCKET.IO CONNECTION
    // ==========================================================================

    function initSocket() {
        if (typeof io === 'undefined') {
            console.warn('Socket.IO not loaded, falling back to REST polling');
            startPolling();
            return;
        }

        socket = io({
            reconnection: true,
            reconnectionAttempts: CONFIG.reconnectAttempts,
            reconnectionDelay: CONFIG.reconnectDelay,
        });

        socket.on('connect', function() {
            console.log('Connected to cockpit server');
            connected = true;
            reconnectAttempts = 0;
            updateConnectionStatus(true);

            // Request current state
            socket.emit('request_state');

            // Request resilience state
            socket.emit('request_resilience');
        });

        socket.on('disconnect', function(reason) {
            console.log('Disconnected:', reason);
            connected = false;
            updateConnectionStatus(false, 'Reconnecting...');
        });

        socket.on('connect_error', function(error) {
            console.error('Connection error:', error);
            reconnectAttempts++;
            updateConnectionStatus(false, 'Connection failed');

            if (reconnectAttempts >= CONFIG.reconnectAttempts) {
                console.log('Max reconnect attempts reached, falling back to REST');
                startPolling();
            }
        });

        // Main state update handler
        socket.on('cockpit_update', function(data) {
            console.log('Received state update:', data.phase);
            handleStateUpdate(data);
        });

        // Resilience state update handler
        socket.on('resilience_update', function(data) {
            console.log('Received resilience update');
            updateResiliencePanel(data);
        });

        // Run changed handler - new run detected
        socket.on('run_changed', function(data) {
            console.log('New run detected:', data.run_id);
            // Clear old logs when switching to new run
            clearOutputPanel();
            // Reset node states to pending
            resetAllNodesToPending();
            // Show notification
            showNotification('New pipeline run: ' + data.run_id);
            // Request fresh state
            socket.emit('request_state');
        });
    }

    function clearOutputPanel() {
        const content = document.getElementById('pipeline-output');
        if (content) {
            content.innerHTML = '';
        }
        // Reset tracking sets
        shownEventIds.clear();
        shownRunLogLines.clear();
        shownRedisEventIds.clear();
    }

    function resetAllNodesToPending() {
        const nodes = document.querySelectorAll('.node');
        nodes.forEach(function(node) {
            node.classList.remove('active', 'complete', 'warning', 'error', 'rejected');
            node.classList.add('pending');
        });
    }

    function showNotification(message) {
        // Update status indicator
        const statusEl = document.getElementById('header-status');
        if (statusEl) {
            statusEl.textContent = message;
            setTimeout(function() {
                statusEl.textContent = 'Running';
            }, 3000);
        }
    }

    // ==========================================================================
    // REST POLLING FALLBACK
    // ==========================================================================

    let pollingInterval = null;

    function startPolling() {
        if (pollingInterval) return;

        console.log('Starting REST polling fallback');
        updateConnectionStatus(false, 'Polling');

        // Initial fetch
        fetchState();

        // Poll every N seconds
        pollingInterval = setInterval(fetchState, CONFIG.pollInterval);
    }

    function stopPolling() {
        if (pollingInterval) {
            clearInterval(pollingInterval);
            pollingInterval = null;
        }
    }

    function fetchState() {
        fetch('/api/cockpit_state')
            .then(function(response) {
                return response.json();
            })
            .then(function(data) {
                if (data.nodes) {
                    handleStateUpdate(data);
                }
            })
            .catch(function(error) {
                console.error('Failed to fetch state:', error);
            });
    }

    // ==========================================================================
    // STATE UPDATE HANDLERS
    // ==========================================================================

    function handleStateUpdate(data) {
        // Handle IDLE/empty state (after reset)
        if (data.phase === 'IDLE' || !data.run_id) {
            console.log('Received IDLE state, clearing UI');
            clearOutputPanel();
            resetAllNodesToPending();
            updateHeaderStats(data);
            return;
        }

        // Update header stats
        updateHeaderStats(data);

        // Update all nodes
        if (data.nodes && Object.keys(data.nodes).length > 0) {
            updateAllNodes(data.nodes);
        }

        // Update summary
        if (data.summary) {
            updateSummary(data.summary);
        }

        // Update timestamp
        if (data.updated_at) {
            updateTimestamp(data.updated_at);
        }

        // Update event logs in output panel (for adopted pipelines)
        if (data.event_logs && data.event_logs.length > 0) {
            updateEventLogs(data.event_logs);
        }

        // Update run log (raw log lines from run.log file)
        if (data.run_log && data.run_log.length > 0) {
            updateRunLog(data.run_log);
        }

        // Update Redis history (real-time events from Redis)
        if (data.redis_history && data.redis_history.length > 0) {
            updateRedisHistory(data.redis_history);
        }

        // Update resilience panel if data is included
        if (data.resilience) {
            updateResiliencePanel(data.resilience);
        }

        // Redraw connections (status might have changed)
        if (typeof drawFlowConnections === 'function') {
            drawFlowConnections();
        }
    }

    function updateEventLogs(eventLogs) {
        const content = document.getElementById('pipeline-output');
        if (!content) return;

        // Filter to only show new events
        const newEvents = eventLogs.filter(e => {
            const eventId = e.timestamp + ':' + e.type;
            if (shownEventIds.has(eventId)) return false;
            shownEventIds.add(eventId);
            return true;
        });

        if (newEvents.length === 0) return;

        // Add new event lines
        newEvents.forEach(event => {
            const lineEl = document.createElement('div');
            lineEl.className = 'log-line';

            // Color code based on level
            if (event.level === 'error') {
                lineEl.className += ' error';
            } else if (event.level === 'warning') {
                lineEl.className += ' warning';
            } else if (event.level === 'success') {
                lineEl.className += ' success';
            } else {
                lineEl.className += ' info';
            }

            // Format: [PHASE] message
            lineEl.textContent = event.line || event.message || JSON.stringify(event);
            content.appendChild(lineEl);
        });

        // Auto-scroll to bottom
        content.scrollTop = content.scrollHeight;
    }

    function updateRunLog(runLogLines) {
        const content = document.getElementById('pipeline-output');
        if (!content) return;

        // Filter to only show new lines
        const newLines = runLogLines.filter(line => {
            if (shownRunLogLines.has(line)) return false;
            shownRunLogLines.add(line);
            return true;
        });

        if (newLines.length === 0) return;

        // Add new log lines
        newLines.forEach(line => {
            const lineEl = document.createElement('div');
            lineEl.className = 'log-line';

            // Color code based on content
            if (line.includes(' - ERROR - ') || line.includes('FAILED') || line.includes('Error:')) {
                lineEl.className += ' error';
            } else if (line.includes(' - WARNING - ') || line.includes('WARN')) {
                lineEl.className += ' warning';
            } else if (line.includes('PASSED') || line.includes('SUCCESS') || line.includes('completed')) {
                lineEl.className += ' success';
            } else {
                lineEl.className += ' info';
            }

            lineEl.textContent = line;
            content.appendChild(lineEl);
        });

        // Auto-scroll to bottom
        content.scrollTop = content.scrollHeight;
    }

    function updateRedisHistory(historyEvents) {
        const content = document.getElementById('pipeline-output');
        if (!content) return;

        // Redis history comes in newest-first order, so reverse for display
        const orderedEvents = [...historyEvents].reverse();

        // Filter to only show new events
        const newEvents = orderedEvents.filter(e => {
            const eventId = e.timestamp + ':' + e.type;
            if (shownRedisEventIds.has(eventId)) return false;
            shownRedisEventIds.add(eventId);
            return true;
        });

        if (newEvents.length === 0) return;

        // Add new event lines
        newEvents.forEach(event => {
            const lineEl = document.createElement('div');
            lineEl.className = 'log-line';

            // Color code based on type/message
            const type = (event.type || '').toUpperCase();
            const msg = event.message || '';

            if (type.includes('FAIL') || type.includes('ERROR') || msg.includes('FAILED')) {
                lineEl.className += ' error';
            } else if (type.includes('WARN')) {
                lineEl.className += ' warning';
            } else if (type.includes('SUCCESS') || type.includes('COMPLETE') || msg.includes('SUCCESS')) {
                lineEl.className += ' success';
            } else {
                lineEl.className += ' info';
            }

            // Format: [TYPE] message (timestamp)
            const timestamp = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : '';
            lineEl.textContent = `[${type}] ${msg}` + (timestamp ? ` (${timestamp})` : '');
            content.appendChild(lineEl);
        });

        // Auto-scroll to bottom
        content.scrollTop = content.scrollHeight;
    }

    function updateHeaderStats(data) {
        // Update phase
        const phaseEl = document.getElementById('header-phase');
        if (phaseEl && data.phase) {
            phaseEl.textContent = data.phase;
        }

        // Update sprint
        const sprintEl = document.getElementById('header-sprint');
        if (sprintEl && data.sprint_id) {
            sprintEl.textContent = data.sprint_id;
        }

        // Update status
        const statusEl = document.getElementById('header-status');
        if (statusEl) {
            statusEl.textContent = data.phase === 'DONE' ? 'Complete' : 'Running';
        }

        // Update metric sprint
        const metricSprintEl = document.getElementById('metric-sprint');
        if (metricSprintEl && data.sprint_id) {
            metricSprintEl.textContent = data.sprint_id;
        }

        // Update progress
        if (data.summary) {
            const total = data.summary.total_nodes || 0;
            const complete = data.summary.complete_nodes || 0;
            const active = data.summary.active_nodes || 0;
            const errors = data.summary.error_nodes || 0;

            const pct = total > 0 ? Math.round((complete / total) * 100) : 0;

            const progressEl = document.getElementById('metric-progress');
            if (progressEl) {
                progressEl.textContent = pct + '%';
            }

            // Update donut chart
            const donutPctEl = document.getElementById('donut-pct');
            if (donutPctEl) {
                donutPctEl.textContent = pct + '%';
            }

            const pctDone = total > 0 ? Math.round((complete / total) * 100) : 0;
            const pctActive = total > 0 ? Math.round((active / total) * 100) : 0;
            const pctPending = 100 - pctDone - pctActive;

            document.getElementById('pct-done') && (document.getElementById('pct-done').textContent = 'Done ' + pctDone + '%');
            document.getElementById('pct-active') && (document.getElementById('pct-active').textContent = 'Active ' + pctActive + '%');
            document.getElementById('pct-pending') && (document.getElementById('pct-pending').textContent = 'Pending ' + pctPending + '%');
        }
    }

    function updateAllNodes(nodes) {
        for (const nodeId in nodes) {
            if (nodes.hasOwnProperty(nodeId)) {
                updateNodeVisual(nodeId, nodes[nodeId]);
                updateNodeData(nodeId, nodes[nodeId]);
            }
        }
    }

    function updateNodeVisual(nodeId, state) {
        const el = document.getElementById('node-' + nodeId);
        if (!el) return;

        // Remove old status classes
        el.classList.remove('pending', 'active', 'complete', 'warning', 'error', 'rejected');

        // Add new status class
        if (state.status) {
            el.classList.add(state.status);
        }
    }

    function updateNodeData(nodeId, state) {
        // Initialize if not exists
        if (!window.nodeData[nodeId]) {
            window.nodeData[nodeId] = {
                title: state.title || nodeId,
                status: 'pending',
                description: '',
                metrics: {},
                substeps: [],
                agents: [],
                logs: [],
            };
        }

        const data = window.nodeData[nodeId];

        // Update status
        if (state.status) {
            data.status = state.status;
        }

        // Update title
        if (state.title) {
            data.title = state.title;
        }

        // Update description
        if (state.description !== undefined) {
            data.description = state.description;
        }

        // Merge metrics
        if (state.metrics) {
            data.metrics = Object.assign({}, data.metrics, state.metrics);
        }

        // Update substeps (replace entirely with latest from backend)
        if (state.substeps !== undefined) {
            data.substeps = state.substeps;
        }

        // Update agents (replace entirely with latest from backend)
        if (state.agents !== undefined) {
            data.agents = state.agents;
        }

        // Update logs (replace with latest from backend, already filtered and sorted)
        if (state.logs !== undefined) {
            data.logs = state.logs;
        }
    }

    function updateSummary(summary) {
        // Update summary panel
        const totalEl = document.getElementById('summary-total');
        if (totalEl && summary.total_nodes !== undefined) {
            totalEl.textContent = summary.total_nodes;
        }

        const completeEl = document.getElementById('summary-complete');
        if (completeEl && summary.complete_nodes !== undefined) {
            completeEl.textContent = summary.complete_nodes;
        }

        const activeEl = document.getElementById('summary-active');
        if (activeEl && summary.active_nodes !== undefined) {
            activeEl.textContent = summary.active_nodes;
        }

        const errorsEl = document.getElementById('summary-errors');
        if (errorsEl && summary.error_nodes !== undefined) {
            errorsEl.textContent = summary.error_nodes;
        }

        const gatesPassedEl = document.getElementById('summary-gates-passed');
        if (gatesPassedEl && summary.gates_passed !== undefined) {
            gatesPassedEl.textContent = summary.gates_passed + '/9';
        }

        const gatesFailedEl = document.getElementById('summary-gates-failed');
        if (gatesFailedEl && summary.gates_failed !== undefined) {
            gatesFailedEl.textContent = summary.gates_failed;
        }

        // Update badge
        const badgeEl = document.getElementById('summary-badge');
        if (badgeEl && summary.total_nodes !== undefined) {
            const pct = summary.total_nodes > 0 ? Math.round((summary.complete_nodes / summary.total_nodes) * 100) : 0;
            badgeEl.textContent = pct + '% Complete';
        }

        // Update warnings metric
        const warningsEl = document.getElementById('metric-warnings');
        if (warningsEl && summary.error_nodes !== undefined) {
            warningsEl.textContent = summary.error_nodes;
        }
    }

    function updateTimestamp(updatedAt) {
        const el = document.getElementById('summary-updated');
        if (!el || !updatedAt) return;

        try {
            const date = new Date(updatedAt);
            el.textContent = date.toLocaleTimeString();
        } catch (e) {
            el.textContent = updatedAt;
        }
    }

    // ==========================================================================
    // NODE POPUP
    // ==========================================================================

    window.showNodePopup = function(nodeId) {
        const data = window.nodeData[nodeId];
        if (!data) {
            console.warn('No data for node:', nodeId);
            return;
        }

        const overlay = document.getElementById('node-popup-overlay');
        const title = document.getElementById('popup-title');
        const body = document.getElementById('popup-body');
        const statusIndicator = overlay.querySelector('.status-indicator');

        if (!overlay || !title || !body) return;

        title.textContent = data.title || nodeId;
        statusIndicator.className = 'status-indicator ' + (data.status || 'pending');

        let html = '';

        // Warning/error banner
        if (data.status === 'warning') {
            html += '<div style="background:rgba(245,158,11,0.15);border:1px solid var(--warning);border-radius:8px;padding:12px;margin-bottom:16px;display:flex;align-items:center;gap:10px;">' +
                '<span style="font-size:20px;">Warning</span>' +
                '<div>' +
                '<div style="color:var(--warning);font-weight:600;font-size:12px;">GUARDRAIL TRIGGERED</div>' +
                '<div style="color:var(--text-secondary);font-size:11px;">This node activated a guardrail and requires attention.</div>' +
                '</div></div>';
        } else if (data.status === 'error') {
            html += '<div style="background:rgba(239,68,68,0.15);border:1px solid var(--error);border-radius:8px;padding:12px;margin-bottom:16px;display:flex;align-items:center;gap:10px;">' +
                '<span style="font-size:20px;">Error</span>' +
                '<div>' +
                '<div style="color:var(--error);font-weight:600;font-size:12px;">REWORK REQUIRED</div>' +
                '<div style="color:var(--text-secondary);font-size:11px;">Work was rejected and sent back for corrections.</div>' +
                '</div></div>';
        }

        // Description
        if (data.description) {
            html += '<p style="color:var(--text-secondary);margin-bottom:20px;font-size:12px;">' + data.description + '</p>';
        }

        // Metrics
        if (data.metrics && Object.keys(data.metrics).length > 0) {
            html += '<div class="node-popup-section">' +
                '<div class="node-popup-section-title">Metrics</div>' +
                '<div class="node-popup-metrics">';
            for (var key in data.metrics) {
                if (data.metrics.hasOwnProperty(key)) {
                    html += '<div class="node-popup-metric">' +
                        '<div class="node-popup-metric-value">' + data.metrics[key] + '</div>' +
                        '<div class="node-popup-metric-label">' + key + '</div>' +
                        '</div>';
                }
            }
            html += '</div></div>';
        }

        // Substeps
        if (data.substeps && data.substeps.length > 0) {
            html += '<div class="node-popup-section">' +
                '<div class="node-popup-section-title">Substeps</div>' +
                '<div class="node-popup-substeps">';
            for (var i = 0; i < data.substeps.length; i++) {
                var step = data.substeps[i];
                var checkClass = 'pending';
                var checkContent = '';
                if (step.status === 'done') { checkClass = 'done'; checkContent = 'OK'; }
                else if (step.status === 'active') { checkClass = 'active'; }
                else if (step.status === 'warning') { checkClass = 'warning'; checkContent = '!'; }
                else if (step.status === 'error') { checkClass = 'error'; checkContent = 'X'; }

                html += '<div class="node-popup-substep">' +
                    '<div class="node-popup-substep-check ' + checkClass + '">' + checkContent + '</div>' +
                    '<div class="node-popup-substep-name">' + step.name + '</div>' +
                    '<div class="node-popup-substep-time">' + (step.time || '-') + '</div>' +
                    '</div>';
            }
            html += '</div></div>';
        }

        // Agents
        if (data.agents && data.agents.length > 0) {
            html += '<div class="node-popup-section">' +
                '<div class="node-popup-section-title">Agents Working</div>' +
                '<div class="node-popup-agents">';
            for (var j = 0; j < data.agents.length; j++) {
                var agent = data.agents[j];
                var agentClass = agent.status === 'active' ? '' : 'idle';
                html += '<div class="node-popup-agent ' + agentClass + '">' +
                    '<div class="node-popup-agent-avatar" style="background:' + (agent.color || 'var(--accent)') + '">' + (agent.avatar || 'A') + '</div>' +
                    '<div class="node-popup-agent-info">' +
                    '<div class="node-popup-agent-name">' + agent.name + '</div>' +
                    '<div class="node-popup-agent-task">' + (agent.task || '') + '</div>' +
                    '</div>' +
                    '<div class="node-popup-agent-status">' +
                    (agent.status === 'active' ? '<span class="dot"></span>' : '') +
                    (agent.status || 'idle') +
                    '</div></div>';
            }
            html += '</div></div>';
        }

        // Logs
        if (data.logs && data.logs.length > 0) {
            html += '<div class="node-popup-section">' +
                '<div class="node-popup-section-title">Activity Log</div>' +
                '<div class="node-popup-log">';
            for (var k = 0; k < data.logs.length; k++) {
                var log = data.logs[k];
                html += '<div class="node-popup-log-entry">' +
                    '<span class="node-popup-log-time">' + (log.time || '') + '</span>' +
                    '<span class="node-popup-log-level ' + (log.level || 'info') + '">' + (log.level || 'INFO').toUpperCase() + '</span>' +
                    '<span class="node-popup-log-msg">' + (log.msg || '') + '</span>' +
                    '</div>';
            }
            html += '</div></div>';
        }

        body.innerHTML = html;
        overlay.classList.add('visible');
    };

    window.closeNodePopup = function(event) {
        if (event && event.target !== event.currentTarget) return;
        var overlay = document.getElementById('node-popup-overlay');
        if (overlay) {
            overlay.classList.remove('visible');
        }
    };

    // ==========================================================================
    // FLOW CONNECTIONS
    // ==========================================================================

    window.drawFlowConnections = function() {
        const svg = document.getElementById('flow-svg');
        const container = document.getElementById('flowchart');
        if (!svg || !container) return;

        const rect = container.getBoundingClientRect();

        function getNodeCenter(id) {
            const node = document.getElementById(id);
            if (!node) return {x: 0, y: 0};
            const nodeRect = node.getBoundingClientRect();
            return {
                x: nodeRect.left - rect.left + nodeRect.width / 2,
                y: nodeRect.top - rect.top + nodeRect.height / 2 - 10
            };
        }

        // Get all node positions
        const nodeIds = [
            'init', 'load', 'spec', 'inv', 'deps', 'plan', 'spawn', 'assign',
            'exec1', 'exec2', 'exec3', 'exec4', 'test',
            'g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8',
            'reflex', 'merge', 'qa', 'review', 'rework',
            'docs', 'art', 'metrics', 'signoff', 'handoff', 'done'
        ];

        const nodes = {};
        nodeIds.forEach(function(id) {
            nodes[id] = getNodeCenter('node-' + id);
        });

        // Get connection states from node data
        function getConnectionState(fromId, toId) {
            const fromData = window.nodeData[fromId];
            const toData = window.nodeData[toId];

            if (!fromData || !toData) return 'pending';

            // If target is active/warning/error, show animated line
            if (toData.status === 'active') return 'active';
            if (toData.status === 'warning') return 'warning';
            if (toData.status === 'error' || toData.status === 'rejected') return 'rejected';
            if (fromData.status === 'complete' && toData.status === 'complete') return 'complete';

            return 'pending';
        }

        // Define connections
        const connections = [
            // INIT → SPEC
            { from: 'init', to: 'load' },
            { from: 'load', to: 'plan' },
            // SPEC Phase (Spec VP → Spec Master → Workers)
            { from: 'plan', to: 'spec' },
            { from: 'spec', to: 'inv' },
            { from: 'spec', to: 'deps' },
            // SPEC → PLAN
            { from: 'inv', to: 'execvp' },
            { from: 'deps', to: 'execvp' },
            // PLAN Phase (Exec VP → Sprint Planner)
            { from: 'execvp', to: 'spawn' },
            // PLAN → EXEC
            { from: 'spawn', to: 'test' },
            // EXEC Phase (Ace Exec → Squad Lead → Workers)
            { from: 'test', to: 'assign' },
            { from: 'assign', to: 'exec1' },
            { from: 'assign', to: 'exec2' },
            { from: 'assign', to: 'exec3' },
            { from: 'assign', to: 'exec4' },
            // EXEC → QA
            { from: 'exec1', to: 'qa' },
            { from: 'exec2', to: 'qa' },
            { from: 'exec3', to: 'qa' },
            { from: 'exec4', to: 'qa' },
            // QA Phase (QA Master → Gates)
            { from: 'qa', to: 'g0' },
            { from: 'g0', to: 'g1' },
            { from: 'g1', to: 'g2' },
            { from: 'g0', to: 'g3' },
            { from: 'g3', to: 'g4' },
            { from: 'g4', to: 'g5' },
            { from: 'g3', to: 'g6' },
            { from: 'g6', to: 'g7' },
            { from: 'g7', to: 'g8' },
            // Gates → Reflexion (on failure)
            { from: 'g8', to: 'reflex' },
            // QA → VOTE
            { from: 'g2', to: 'merge' },
            { from: 'g5', to: 'merge' },
            { from: 'g8', to: 'merge' },
            { from: 'reflex', to: 'merge' },
            // VOTE Phase
            { from: 'merge', to: 'review' },
            { from: 'review', to: 'rework' },
            { from: 'rework', to: 'test' },  // Rework loop back to EXEC
            // VOTE → SIGNOFF
            { from: 'review', to: 'docs' },
            { from: 'review', to: 'art' },
            { from: 'review', to: 'metrics' },
            // SIGNOFF Phase
            { from: 'docs', to: 'signoff' },
            { from: 'art', to: 'signoff' },
            { from: 'metrics', to: 'signoff' },
            // Final
            { from: 'signoff', to: 'handoff' },
            { from: 'handoff', to: 'done' }
        ];

        svg.setAttribute('width', rect.width);
        svg.setAttribute('height', rect.height);

        let svgContent = '<defs>' +
            '<marker id="arrow-complete" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">' +
            '<polygon points="0 0, 10 3.5, 0 7" fill="var(--success)" /></marker>' +
            '<marker id="arrow-active" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">' +
            '<polygon points="0 0, 10 3.5, 0 7" fill="var(--accent)" /></marker>' +
            '<marker id="arrow-pending" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">' +
            '<polygon points="0 0, 10 3.5, 0 7" fill="var(--border-default)" opacity="0.5"/></marker>' +
            '<marker id="arrow-rejected" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">' +
            '<polygon points="0 0, 10 3.5, 0 7" fill="var(--error)" /></marker>' +
            '<marker id="arrow-warning" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">' +
            '<polygon points="0 0, 10 3.5, 0 7" fill="var(--warning)" /></marker>' +
            '</defs>';

        connections.forEach(function(conn) {
            const from = nodes[conn.from];
            const to = nodes[conn.to];
            if (!from || !to || (from.x === 0 && from.y === 0) || (to.x === 0 && to.y === 0)) return;

            const state = getConnectionState(conn.from, conn.to);

            // Calculate curved path
            const midX = (from.x + to.x) / 2;
            const midY = (from.y + to.y) / 2;
            const dx = to.x - from.x;
            const dy = to.y - from.y;
            const offset = Math.abs(dy) > 20 ? dx * 0.2 : 0;

            const path = 'M ' + from.x + ' ' + from.y + ' Q ' + (midX + offset) + ' ' + midY + ' ' + to.x + ' ' + to.y;

            // Only show arrows for active states
            const showArrow = ['active', 'warning', 'rejected'].indexOf(state) >= 0;
            const arrowMarker = showArrow ? 'marker-end="url(#arrow-' + state + ')"' : '';

            svgContent += '<path class="flow-line ' + state + '" d="' + path + '" ' + arrowMarker + ' />';
        });

        svg.innerHTML = svgContent;
    };

    // ==========================================================================
    // FULLSCREEN
    // ==========================================================================

    window.toggleFlowFullscreen = function() {
        const panel = document.getElementById('flow-panel');
        if (panel) {
            panel.classList.toggle('fullscreen');
            setTimeout(drawFlowConnections, 100);
        }
    };

    // ==========================================================================
    // KEYBOARD SHORTCUTS
    // ==========================================================================

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            const panel = document.getElementById('flow-panel');
            if (panel) {
                panel.classList.remove('fullscreen');
            }
            closeNodePopup();
            setTimeout(drawFlowConnections, 100);
        }
    });

    // ==========================================================================
    // RESILIENCE PANEL FUNCTIONS
    // ==========================================================================

    /**
     * Toggle resilience panel collapse state.
     */
    window.toggleResiliencePanel = function() {
        const panel = document.getElementById('resilience-panel');
        if (panel) {
            panel.classList.toggle('collapsed');
        }
    };

    /**
     * Update circuit breaker cards in the UI.
     * @param {Array} cards - Array of circuit breaker card objects
     */
    function updateCircuitBreakerCards(cards) {
        const container = document.getElementById('circuit-breakers-container');
        if (!container) return;

        if (!cards || cards.length === 0) {
            container.innerHTML = '<div class="cb-card-placeholder"><span>No circuit breakers registered</span></div>';
            return;
        }

        container.innerHTML = cards.map(function(card) {
            const cssClass = card.css_class || 'cb-unknown';
            const name = escapeHtml(card.name || 'Unknown');
            const state = card.state || 'UNKNOWN';
            const failureCount = card.failure_count || 0;
            const successCount = card.success_count || 0;

            return '<div class="cb-card ' + cssClass + '" title="' + name + '">' +
                '<div class="cb-card-name">' + name + '</div>' +
                '<div class="cb-card-state"><span class="cb-state-dot"></span><span>' + state + '</span></div>' +
                '<div class="cb-card-stats">F:' + failureCount + ' S:' + successCount + '</div>' +
                '</div>';
        }).join('');
    }

    /**
     * Update oscillation alert banner.
     * @param {Object} oscillation - Oscillation state object
     */
    function updateOscillationAlert(oscillation) {
        const alert = document.getElementById('oscillation-alert');
        const title = document.getElementById('oscillation-title');
        const message = document.getElementById('oscillation-message');

        if (!alert || !title || !message) return;

        if (!oscillation || !oscillation.has_alert) {
            alert.style.display = 'none';
            return;
        }

        alert.style.display = 'flex';
        alert.className = 'oscillation-alert alert-' + (oscillation.alert_level || 'info');

        var patterns = oscillation.active_patterns || [];
        var patternText = patterns.length > 0 ? patterns.join(', ') : 'Unknown';
        title.textContent = (oscillation.alert_level || 'ALERT').toUpperCase() + ': ' + patternText;
        message.textContent = oscillation.alert_message || 'Oscillation pattern detected';
    }

    /**
     * Dismiss oscillation alert.
     */
    window.dismissOscillationAlert = function() {
        const alert = document.getElementById('oscillation-alert');
        if (alert) {
            alert.style.display = 'none';
        }
    };

    /**
     * Update retry metrics display.
     * @param {Object} metrics - Retry metrics object
     */
    function updateRetryMetrics(metrics) {
        const attempts = document.getElementById('retry-attempts');
        const successes = document.getElementById('retry-successes');
        const failures = document.getElementById('retry-failures');
        const rate = document.getElementById('retry-rate');
        const progress = document.getElementById('retry-progress');

        var totalAttempts = (metrics && metrics.total_attempts) || 0;
        var totalSuccesses = (metrics && metrics.total_successes) || 0;
        var totalFailures = (metrics && metrics.total_failures) || 0;
        var successRate = (metrics && metrics.success_rate) || 0;

        if (attempts) attempts.textContent = totalAttempts.toLocaleString();
        if (successes) successes.textContent = totalSuccesses.toLocaleString();
        if (failures) failures.textContent = totalFailures.toLocaleString();
        if (rate) rate.textContent = successRate.toFixed(1) + '%';
        if (progress) progress.style.width = successRate + '%';
    }

    /**
     * Update overall health indicator.
     * @param {boolean} healthy - Whether the system is healthy
     */
    function updateOverallHealth(healthy) {
        const indicator = document.getElementById('resilience-health');
        if (indicator) {
            indicator.className = 'health-indicator ' + (healthy !== false ? 'health-good' : 'health-bad');
            indicator.title = healthy !== false ? 'System Healthy' : 'System Unhealthy';
        }
    }

    /**
     * Main resilience panel update function.
     * @param {Object} data - Resilience state data from server
     */
    function updateResiliencePanel(data) {
        const unavailable = document.getElementById('resilience-unavailable');
        const sections = document.querySelectorAll('.resilience-section');

        // Handle unavailable state
        if (!data || !data.available) {
            if (unavailable) unavailable.style.display = 'flex';
            sections.forEach(function(el) { el.style.display = 'none'; });
            updateOverallHealth(true);  // Default to healthy when unavailable
            return;
        }

        // Show sections, hide unavailable notice
        if (unavailable) unavailable.style.display = 'none';
        sections.forEach(function(el) { el.style.display = 'block'; });

        // Update all components
        updateCircuitBreakerCards(data.circuit_breakers || []);
        updateOscillationAlert(data.oscillation || {});
        updateRetryMetrics(data.retry_metrics || {});
        updateOverallHealth(data.overall_health !== false);
    }

    // ==========================================================================
    // INITIALIZATION
    // ==========================================================================

    function init() {
        // Connection status is already in HTML, just update it
        updateConnectionStatus(false, 'Connecting...');

        // Initialize socket connection
        initSocket();

        // Draw initial connections
        setTimeout(drawFlowConnections, 200);

        // Redraw on resize
        window.addEventListener('resize', function() {
            if (typeof drawFlowConnections === 'function') {
                drawFlowConnections();
            }
        });

        // Start elapsed timer
        startElapsedTimer();

        console.log('Cockpit initialized');
    }

    // Elapsed time counter
    let elapsedSeconds = 0;
    let elapsedInterval = null;

    function startElapsedTimer() {
        if (elapsedInterval) return;

        elapsedInterval = setInterval(function() {
            elapsedSeconds++;
            const h = Math.floor(elapsedSeconds / 3600);
            const m = Math.floor((elapsedSeconds % 3600) / 60);
            const s = elapsedSeconds % 60;
            const el = document.getElementById('elapsed');
            if (el) {
                el.textContent = String(h).padStart(2, '0') + ':' +
                                 String(m).padStart(2, '0') + ':' +
                                 String(s).padStart(2, '0');
            }
        }, 1000);
    }

    // ==========================================================================
    // PIPELINE CONTROL
    // ==========================================================================

    let pipelineOutputLines = [];

    function initPipelineControl() {
        // Load available sprints
        fetch('/api/pipeline/sprints')
            .then(r => r.json())
            .then(data => {
                populateSprintSelects(data.sprints);
                populatePackShortcuts(data.packs);
            })
            .catch(err => console.error('Failed to load sprints:', err));

        // Load initial pipeline status
        refreshPipelineStatus();

        // Listen for pipeline events via WebSocket
        if (socket) {
            socket.on('pipeline_status', handlePipelineStatus);
            socket.on('pipeline_output', handlePipelineOutput);
        }
    }

    function populateSprintSelects(sprints) {
        const startSelect = document.getElementById('start-sprint');
        const endSelect = document.getElementById('end-sprint');

        if (!startSelect || !endSelect) return;

        startSelect.innerHTML = '';
        endSelect.innerHTML = '';

        sprints.forEach(sprint => {
            const opt1 = document.createElement('option');
            opt1.value = sprint;
            opt1.textContent = sprint;
            startSelect.appendChild(opt1);

            const opt2 = document.createElement('option');
            opt2.value = sprint;
            opt2.textContent = sprint;
            endSelect.appendChild(opt2);
        });

        // Set defaults
        startSelect.value = 'S00';
        endSelect.value = 'S25';
    }

    function populatePackShortcuts(packs) {
        const container = document.getElementById('pack-shortcuts');
        if (!container) return;

        container.innerHTML = '';

        packs.forEach((pack, idx) => {
            const btn = document.createElement('button');
            btn.className = 'pack-shortcut';
            btn.textContent = 'P' + (idx + 1);
            btn.title = pack.name;
            btn.onclick = function() {
                document.getElementById('start-sprint').value = pack.start;
                document.getElementById('end-sprint').value = pack.end;
            };
            container.appendChild(btn);
        });
    }

    function refreshPipelineStatus() {
        fetch('/api/pipeline/status')
            .then(r => r.json())
            .then(handlePipelineStatus)
            .catch(err => console.error('Failed to get pipeline status:', err));
    }

    function handlePipelineStatus(status) {
        const indicator = document.getElementById('pipeline-status-indicator');
        const text = document.getElementById('pipeline-status-text');
        const btnStart = document.getElementById('btn-start');
        const btnPause = document.getElementById('btn-pause');
        const btnResume = document.getElementById('btn-resume');
        const btnAbort = document.getElementById('btn-abort');
        const outputDot = document.getElementById('output-status-dot');

        if (!indicator || !text) return;

        // Update indicator
        indicator.className = 'status-indicator ' + status.status;

        // Update text
        let statusText = status.status.charAt(0).toUpperCase() + status.status.slice(1);
        if (status.start_sprint && status.end_sprint) {
            statusText += ` (${status.start_sprint}→${status.end_sprint})`;
        }
        if (status.adopted) {
            statusText += ' [adopted]';
        }
        text.textContent = statusText;

        // Update buttons based on status
        const isRunning = status.status === 'running';
        const isPaused = status.status === 'paused';
        const isActive = isRunning || isPaused;

        if (btnStart) btnStart.disabled = isActive;
        if (btnPause) btnPause.disabled = !isRunning;
        if (btnResume) btnResume.disabled = !isPaused;
        if (btnAbort) btnAbort.disabled = !isActive;

        // Update output panel status dot
        if (outputDot) {
            outputDot.className = 'status-dot' + (isRunning ? ' running' : '');
        }

        // Show output console if pipeline is active
        if (isActive && status.recent_output && status.recent_output.length > 0) {
            showPipelineOutput(status.recent_output);
        }

        // Show message for adopted pipelines
        if (status.adopted && isActive) {
            showAdoptedPipelineMessage(status);
        }
    }

    function showAdoptedPipelineMessage(status) {
        const content = document.getElementById('pipeline-output');
        if (!content) return;

        // Only show once
        if (content.querySelector('.adopted-message')) return;

        const msgEl = document.createElement('div');
        msgEl.className = 'log-line warning adopted-message';
        msgEl.textContent = `[ADOPTED] Pipeline PID ${status.pid} was started externally. Live output not available. Showing event logs instead.`;
        content.insertBefore(msgEl, content.firstChild);
    }

    function handlePipelineOutput(data) {
        pipelineOutputLines.push(data.line);
        if (pipelineOutputLines.length > 500) {
            pipelineOutputLines = pipelineOutputLines.slice(-250);
        }

        const content = document.getElementById('pipeline-output');
        if (content) {
            const lineEl = document.createElement('div');
            lineEl.className = 'log-line';

            // Color code based on content
            if (data.line.includes('ERROR') || data.line.includes('FAIL') || data.line.includes('Error')) {
                lineEl.className += ' error';
            } else if (data.line.includes('SUCCESS') || data.line.includes('PASS') || data.line.includes('✓')) {
                lineEl.className += ' success';
            } else if (data.line.includes('WARN') || data.line.includes('Warning')) {
                lineEl.className += ' warning';
            } else {
                lineEl.className += ' info';
            }

            lineEl.textContent = data.line;
            content.appendChild(lineEl);

            // Auto-scroll to bottom
            content.scrollTop = content.scrollHeight;
        }

        // Make sure output panel is visible
        const outputPanel = document.getElementById('pipeline-output');
        if (outputPanel && !outputPanel.classList.contains('visible')) {
            outputPanel.classList.add('visible');
        }
    }

    function showPipelineOutput(lines) {
        const outputPanel = document.getElementById('pipeline-output');
        if (!outputPanel || !lines) return;

        // Show recent output lines
        outputPanel.innerHTML = lines.map(line => {
            let cls = 'log-line';
            if (line.includes('ERROR') || line.includes('FAIL') || line.includes('Error')) {
                cls += ' error';
            } else if (line.includes('SUCCESS') || line.includes('PASS') || line.includes('✓')) {
                cls += ' success';
            } else if (line.includes('WARN') || line.includes('Warning')) {
                cls += ' warning';
            } else {
                cls += ' info';
            }
            return `<div class="${cls}">${escapeHtml(line)}</div>`;
        }).join('');

        // Auto-scroll to bottom
        outputPanel.scrollTop = outputPanel.scrollHeight;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Global functions for button onclick handlers
    window.pipelineStart = function() {
        const startSprint = document.getElementById('start-sprint').value;
        const endSprint = document.getElementById('end-sprint').value;

        fetch('/api/pipeline/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                start_sprint: startSprint,
                end_sprint: endSprint,
            }),
        })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    console.log('Pipeline started:', data);
                    handlePipelineStatus(data.status);
                } else {
                    alert('Failed to start pipeline: ' + data.error);
                }
            })
            .catch(err => {
                console.error('Error starting pipeline:', err);
                alert('Error starting pipeline: ' + err.message);
            });
    };

    window.pipelinePause = function() {
        fetch('/api/pipeline/pause', { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    console.log('Pipeline paused');
                    handlePipelineStatus(data.status);
                } else {
                    alert('Failed to pause pipeline: ' + data.error);
                }
            })
            .catch(err => {
                console.error('Error pausing pipeline:', err);
                alert('Error pausing pipeline: ' + err.message);
            });
    };

    window.pipelineResume = function() {
        fetch('/api/pipeline/resume', { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    console.log('Pipeline resumed');
                    handlePipelineStatus(data.status);
                } else {
                    alert('Failed to resume pipeline: ' + data.error);
                }
            })
            .catch(err => {
                console.error('Error resuming pipeline:', err);
                alert('Error resuming pipeline: ' + err.message);
            });
    };

    window.pipelineAbort = function() {
        if (!confirm('Are you sure you want to abort the pipeline?')) {
            return;
        }

        fetch('/api/pipeline/abort', { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    console.log('Pipeline aborted');
                    handlePipelineStatus(data.status);
                } else {
                    alert('Failed to abort pipeline: ' + data.error);
                }
            })
            .catch(err => {
                console.error('Error aborting pipeline:', err);
                alert('Error aborting pipeline: ' + err.message);
            });
    };

    // Resume from LangGraph checkpoint
    window.pipelineLgResume = function() {
        // First fetch available checkpoints
        fetch('/api/pipeline/checkpoints')
            .then(r => r.json())
            .then(data => {
                if (data.error && !data.checkpoints?.length) {
                    alert('No checkpoints available: ' + data.error);
                    return;
                }

                if (!data.checkpoints || data.checkpoints.length === 0) {
                    alert('No checkpoints found. Run the pipeline first to create checkpoints.');
                    return;
                }

                // Build a simple selection prompt
                let msg = 'Available checkpoints:\n\n';
                data.checkpoints.forEach((cp, i) => {
                    msg += `${i + 1}. ${cp.id} (${cp.phase})\n`;
                });
                msg += '\nEnter number to resume (or cancel):';

                const selection = prompt(msg, '1');
                if (!selection) return;

                const idx = parseInt(selection) - 1;
                if (idx < 0 || idx >= data.checkpoints.length) {
                    alert('Invalid selection');
                    return;
                }

                const checkpoint = data.checkpoints[idx];

                // Confirm and resume
                if (!confirm(`Resume from checkpoint:\n${checkpoint.id}\nPhase: ${checkpoint.phase}`)) {
                    return;
                }

                fetch('/api/pipeline/lg-resume', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ checkpoint_id: checkpoint.id }),
                })
                    .then(r => r.json())
                    .then(result => {
                        if (result.success) {
                            console.log('Pipeline resumed from checkpoint:', result);
                            handlePipelineStatus(result.status);
                        } else {
                            alert('Failed to resume: ' + result.error);
                        }
                    })
                    .catch(err => {
                        console.error('Error resuming from checkpoint:', err);
                        alert('Error: ' + err.message);
                    });
            })
            .catch(err => {
                console.error('Error fetching checkpoints:', err);
                alert('Error fetching checkpoints: ' + err.message);
            });
    };

    window.hidePipelineOutput = function() {
        const outputPanel = document.getElementById('pipeline-output');
        if (outputPanel) {
            outputPanel.classList.remove('visible');
        }
    };

    window.pipelineReset = function() {
        if (!confirm('Force reset? This will kill ALL pipeline processes.')) {
            return;
        }

        fetch('/api/pipeline/reset', { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                console.log('Pipeline reset:', data);
                handlePipelineStatus(data.status);

                // Clear output panel completely
                clearOutputPanel();

                // Reset all nodes to pending
                resetAllNodesToPending();

                // Update header
                const phaseEl = document.getElementById('header-phase');
                if (phaseEl) phaseEl.textContent = 'IDLE';

                const runIdEl = document.getElementById('header-run-id');
                if (runIdEl) runIdEl.textContent = '-';

                const statusEl = document.getElementById('header-status');
                if (statusEl) statusEl.textContent = 'Ready';

                // Show message
                const content = document.getElementById('pipeline-output');
                if (content) {
                    content.innerHTML = '<div class="log-line info">Pipeline state reset. Ready to start new run.</div>';
                }

                alert('Pipeline reset complete. You can start a new run.');
            })
            .catch(err => {
                console.error('Error resetting pipeline:', err);
                alert('Error resetting pipeline: ' + err.message);
            });
    };

    window.clearOutput = function() {
        const content = document.getElementById('pipeline-output');
        if (content) {
            content.innerHTML = '';
        }
        pipelineOutputLines = [];
    };

    // Run when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            init();
            initPipelineControl();
        });
    } else {
        init();
        initPipelineControl();
    }

})();
