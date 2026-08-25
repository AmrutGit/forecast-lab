export function Header() {
  return (
    <header className="app-header">
      <div className="app-header__brand">
        <span className="app-header__mark" aria-hidden="true" />
        <div>
          <p className="app-header__title">Forecast Lab</p>
          <p className="app-header__subtitle">Regional demand dashboard</p>
        </div>
      </div>
    </header>
  );
}
