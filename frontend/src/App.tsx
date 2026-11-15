type MetricCardProps = {
  label: string;
  value: string;
  delta: string;
};

function MetricCard({ label, value, delta }: MetricCardProps) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-delta">{delta}</div>
    </div>
  );
}

export default function App() {
  const metrics = [
    { label: 'Invoices processed', value: '1,248', delta: '+8.4% vs last month' },
    { label: 'Pending exceptions', value: '36', delta: '-12.1%' },
    { label: 'Duplicate invoices', value: '5', delta: '2 flagged for review' },
    { label: 'Expense violations', value: '11', delta: '+3 this week' },
  ];

  const rows = [
    { vendor: 'ABC Technologies', risk: 'Moderate', status: 'Manual review' },
    { vendor: 'Northwind Office Supply', risk: 'Low', status: 'Approved' },
    { vendor: 'BluePeak Logistics', risk: 'High', status: 'Escalated' },
  ];

  return (
    <div className="page-shell">
      <aside className="sidebar">
        <div className="sidebar-header">FinanceOps AI</div>
        <nav>
          <a href="#">Executive Dashboard</a>
          <a href="#">Invoices</a>
          <a href="#">Expenses</a>
          <a href="#">Vendor Risk</a>
          <a href="#">Anomalies</a>
          <a href="#">Audit</a>
        </nav>
      </aside>
      <main className="content">
        <header className="topbar">
          <div>
            <h1>Operations Copilot</h1>
            <p>Financial performance and control monitoring</p>
          </div>
          <button className="primary-btn">Run audit review</button>
        </header>

        <section className="metrics-grid">
          {metrics.map((metric) => (
            <MetricCard key={metric.label} {...metric} />
          ))}
        </section>

        <section className="panel-grid">
          <div className="panel">
            <h2>AI prompt</h2>
            <div className="prompt-box">
              “Summarize the FY2025 financial performance and identify any high-risk vendor exceptions.”
            </div>
          </div>
          <div className="panel">
            <h2>Risk watchlist</h2>
            <table>
              <thead>
                <tr>
                  <th>Vendor</th>
                  <th>Risk</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.vendor}>
                    <td>{row.vendor}</td>
                    <td>{row.risk}</td>
                    <td>{row.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}
