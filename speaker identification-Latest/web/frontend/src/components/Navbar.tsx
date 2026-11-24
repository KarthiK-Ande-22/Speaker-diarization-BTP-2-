import { Plus, Stethoscope } from "lucide-react";

const LINKS = [
  { page: "home", label: "Home", href: "#/" },
  { page: "recordings", label: "Recordings", href: "#/recordings" },
  { page: "how", label: "How it works", href: "#/how-it-works" },
];

export default function Navbar({ page }: { page: string }) {
  return (
    <header className="navbar">
      <div className="navbar-inner">
        <a className="logo" href="#/">
          <span className="logo-icon">
            <Stethoscope size={18} strokeWidth={2.2} />
          </span>
          Role Identification
        </a>
        <nav className="nav-links">
          {LINKS.map((link) => (
            <a key={link.page} href={link.href} className={`nav-link${page === link.page ? " active" : ""}`}>
              {link.label}
            </a>
          ))}
          <a className="button nav-cta" href="#/">
            <Plus size={16} strokeWidth={2.5} /> New analysis
          </a>
        </nav>
      </div>
    </header>
  );
}
