import { useId, useMemo } from 'react'
import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from 'd3-force'
import type { AnalysisGraphEdge, AnalysisGraphNode, AnalysisPatternFinding, AnalysisPatternType } from '../types/api'

type LayoutNode = AnalysisGraphNode & SimulationNodeDatum
type LayoutLink = Omit<AnalysisGraphEdge, 'source' | 'target'> & SimulationLinkDatum<LayoutNode> & {
  source: string | LayoutNode
  target: string | LayoutNode
}

const patternColors: Record<AnalysisPatternType, string> = {
  cycle: '#a94d4d',
  'fan-in': '#a66b22',
  'fan-out': '#6656a2',
  'layered-chain': '#24736d',
  'rapid-pass-through': '#3978a0',
  'high-velocity': '#8a5d83',
}

const patternLabels: Record<AnalysisPatternType, string> = {
  cycle: 'Circular routing',
  'fan-in': 'Fan-in',
  'fan-out': 'Fan-out',
  'layered-chain': 'Layered chain',
  'rapid-pass-through': 'Rapid pass-through',
  'high-velocity': 'High velocity',
}

type NetworkGraphProps = {
  nodes: AnalysisGraphNode[]
  edges: AnalysisGraphEdge[]
  patterns: AnalysisPatternFinding[]
  selectedPattern: string
  accountSearch: string
  minRisk: number
  maxRisk: number
  onSelectNode: (node: AnalysisGraphNode) => void
  onSelectEdge: (edge: AnalysisGraphEdge) => void
}

export function NetworkGraph({
  nodes,
  edges,
  patterns,
  selectedPattern,
  accountSearch,
  minRisk,
  maxRisk,
  onSelectNode,
  onSelectEdge,
}: NetworkGraphProps) {
  const markerId = `arrow-${useId().replace(/:/g, '')}`
  const layout = useMemo(() => {
    const patternById = new Map(patterns.map((pattern) => [pattern.id, pattern]))
    const edgesForPattern = selectedPattern === 'all'
      ? edges
      : edges.filter((edge) => edge.patternIds.some((id) => patternById.get(id)?.pattern === selectedPattern))
    const query = accountSearch.trim().toLowerCase()
    const riskFiltered = new Set(nodes.filter((node) => {
      if (node.riskScore === null) return minRisk === 0 && maxRisk === 100
      return node.riskScore >= minRisk && node.riskScore <= maxRisk
    }).map((node) => node.id))
    const matchingAccounts = query
      ? new Set(nodes.filter((node) => riskFiltered.has(node.id) && node.id.toLowerCase().includes(query)).map((node) => node.id))
      : null
    let candidateEdges = edgesForPattern.filter((edge) => riskFiltered.has(edge.source) && riskFiltered.has(edge.target))
    if (matchingAccounts) {
      const contextAccounts = new Set(matchingAccounts)
      for (const edge of candidateEdges) {
        if (matchingAccounts.has(edge.source) || matchingAccounts.has(edge.target)) {
          contextAccounts.add(edge.source)
          contextAccounts.add(edge.target)
        }
      }
      candidateEdges = candidateEdges.filter((edge) => contextAccounts.has(edge.source) && contextAccounts.has(edge.target))
    }
    const filteredNodeIds = new Set(candidateEdges.flatMap((edge) => [edge.source, edge.target]))
    if (candidateEdges.length === 0 && matchingAccounts) {
      for (const account of matchingAccounts) filteredNodeIds.add(account)
    }
    const limitedEdges = [...candidateEdges]
      .sort((a, b) => {
        const priorityA = (a.patternIds.length ? 2 : 0) + ((a.riskAssessment.finalScore ?? -1) >= 75 ? 1 : 0)
        const priorityB = (b.patternIds.length ? 2 : 0) + ((b.riskAssessment.finalScore ?? -1) >= 75 ? 1 : 0)
        return priorityB - priorityA || a.timestamp.localeCompare(b.timestamp) || a.id.localeCompare(b.id)
      })
      .slice(0, 1000)
    const usedAccounts = new Set(limitedEdges.flatMap((edge) => [edge.source, edge.target]))
    if (limitedEdges.length === 0 && matchingAccounts) {
      for (const account of matchingAccounts) usedAccounts.add(account)
    }
    const accountById = new Map(nodes.map((node) => [node.id, node]))
    const visibleNodes = [...usedAccounts]
      .map((id) => accountById.get(id))
      .filter((node): node is AnalysisGraphNode => node !== undefined)
      .slice(0, 300)
    const visibleIds = new Set(visibleNodes.map((node) => node.id))
    const visibleEdges = limitedEdges
      .filter((edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target))
      .slice(0, 1000)
    const simulationNodes: LayoutNode[] = visibleNodes.map((node) => ({ ...node }))
    const simulationLinks: LayoutLink[] = visibleEdges.map((edge) => ({ ...edge }))
    const simulation = forceSimulation<LayoutNode>(simulationNodes)
      .force('link', forceLink<LayoutNode, LayoutLink>(simulationLinks).id((node) => node.id).distance(92).strength(0.7))
      .force('charge', forceManyBody<LayoutNode>().strength(-185))
      .force('center', forceCenter<LayoutNode>(480, 270))
      .force('x', forceX<LayoutNode>(480).strength(0.035))
      .force('y', forceY<LayoutNode>(270).strength(0.035))
      .force('collision', forceCollide<LayoutNode>((node) => Math.min(18, 8 + node.outDegree * 0.5)).strength(0.8))
      .stop()
    simulation.tick(180)
    return {
      nodes: simulationNodes,
      links: simulationLinks,
      totalNodes: filteredNodeIds.size,
      totalEdges: candidateEdges.length,
      patternById,
      edgeById: new Map(edges.map((edge) => [edge.id, edge])),
      nodeById: new Map(simulationNodes.map((node) => [node.id, node])),
    }
  }, [accountSearch, edges, maxRisk, minRisk, nodes, patterns, selectedPattern])

  const edgePattern = (edge: Pick<AnalysisGraphEdge, 'patternIds'>) => edge.patternIds
    .map((id) => layout.patternById.get(id)?.pattern)
    .find((pattern): pattern is AnalysisPatternType => pattern !== undefined)
  const nodePattern = (node: LayoutNode) => patterns
    .filter((pattern) => pattern.accountIds.includes(node.id))
    .map((pattern) => pattern.pattern)
    .sort((a, b) => Object.keys(patternColors).indexOf(a) - Object.keys(patternColors).indexOf(b))[0]

  return (
    <div className="network-graph-wrap">
      <div className="network-graph-legend" aria-label="Network graph legend">
        {Object.entries(patternLabels).map(([pattern, label]) => (
          <span key={pattern} className="network-legend-item">
            <i aria-hidden="true" style={{ backgroundColor: patternColors[pattern as AnalysisPatternType] }} />
            {label}
          </span>
        ))}
        <span className="network-legend-item">
          <i aria-hidden="true" className="network-risk-dot" />
          AI risk signal
        </span>
      </div>
      {layout.nodes.length === 0 ? (
        <div className="chart-placeholder">
          <div className="chart-placeholder-inner">
            <h3 className="state-title">No accounts match these filters</h3>
            <p className="state-description">Change the account search, pattern type, or risk range to explore this dataset.</p>
          </div>
        </div>
      ) : (
        <svg
          aria-label="Directed transfer graph. Select an account or transfer to inspect its evidence."
          className="network-graph"
          role="group"
          viewBox="0 0 960 540"
        >
          <defs>
            <marker id={markerId} markerHeight="7" markerWidth="8" orient="auto" refX="17" refY="3.5">
              <path d="M0,0 L0,7 L8,3.5 z" fill="#829496" />
            </marker>
          </defs>
          {layout.links.map((link) => {
            const sourceId = typeof link.source === 'string' ? link.source : link.source.id
            const targetId = typeof link.target === 'string' ? link.target : link.target.id
            const source = layout.nodeById.get(sourceId)
            const target = layout.nodeById.get(targetId)
            const originalEdge = layout.edgeById.get(link.id)
            if (!source || !target) return null
            const sourceX = Math.min(930, Math.max(30, source.x ?? 480))
            const sourceY = Math.min(510, Math.max(30, source.y ?? 270))
            const targetX = Math.min(930, Math.max(30, target.x ?? 480))
            const targetY = Math.min(510, Math.max(30, target.y ?? 270))
            const pattern = edgePattern(link)
            const color = pattern ? patternColors[pattern] : '#829496'
            return (
              <g key={link.id} className="network-edge-group">
                <line
                  aria-label={`${sourceId} to ${targetId}, ${link.amount}`}
                  className="network-edge-hitbox"
                  role="button"
                  tabIndex={0}
                  x1={sourceX}
                  x2={targetX}
                  y1={sourceY}
                  y2={targetY}
                  onClick={() => { if (originalEdge) onSelectEdge(originalEdge) }}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault()
                      if (originalEdge) onSelectEdge(originalEdge)
                    }
                  }}
                />
                <line
                  className={`network-edge${pattern === 'rapid-pass-through' ? ' is-dashed' : ''}`}
                  markerEnd={`url(#${markerId})`}
                  stroke={color}
                  x1={sourceX}
                  x2={targetX}
                  y1={sourceY}
                  y2={targetY}
                />
                <title>{`${sourceId} → ${targetId} · ${link.amount.toLocaleString()} · ${new Date(link.timestamp).toLocaleString()}`}</title>
              </g>
            )
          })}
          {layout.nodes.map((node) => {
            const x = Math.min(930, Math.max(30, node.x ?? 480))
            const y = Math.min(510, Math.max(30, node.y ?? 270))
            const pattern = nodePattern(node)
            const riskColor = node.riskScore === null
              ? '#829496'
              : node.riskScore >= 75 ? '#b95752' : node.riskScore >= 40 ? '#c18b38' : '#528b8e'
            return (
              <g
                key={node.id}
                aria-label={`Account ${node.id}, ${node.transactionCount} transfers, ${node.patternCount} findings`}
                className="network-node"
                role="button"
                tabIndex={0}
                transform={`translate(${x},${y})`}
                onClick={() => onSelectNode(node)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault()
                    onSelectNode(node)
                  }
                }}
              >
                <circle className="network-node-halo" r={node.patternCount ? 14 : 11} />
                <circle
                  className="network-node-core"
                  fill={pattern ? patternColors[pattern] : riskColor}
                  r={Math.min(11, 6 + Math.log2(node.transactionCount + 1))}
                />
                <text className="network-node-label" dy="25" textAnchor="middle">{node.id}</text>
                <title>{`${node.id} · ${node.transactionCount} transfers · ${node.patternCount} pattern(s) · ${node.riskScore === null ? 'AI score unavailable' : `average AI risk ${node.riskScore.toFixed(1)}`}`}</title>
              </g>
            )
          })}
        </svg>
      )}
      <p className="network-graph-note">
        Showing {layout.nodes.length} of {layout.totalNodes} accounts and {layout.links.length} of {layout.totalEdges} transfers
        {layout.totalNodes > 300 || layout.totalEdges > 1000 ? ' (visualization is capped for responsiveness).' : '.'}
        {' '}AI risk color is separate from pattern evidence. Select a node or arrow for details.
      </p>
    </div>
  )
}
