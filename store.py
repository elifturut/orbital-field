"""Sigortacı kararlarının kalıcı kayıt defteri (SQLite; saf Python, Streamlit gerekmez).

Eklemeli (append-only) tek tablo: her karar ya da geri alma yeni bir satırdır; hiçbir satır güncellenmez ya da silinmez.
Bir poliçenin GEÇERLİ kararı, o poliçenin son kaydıdır (:func:`casework.effective`). Böylece "kim, ne zaman, neyi
değiştirdi" izi her zaman korunur.

Dürüst sınır
------------
* Dosya sunucudaki yerel bir SQLite dosyasıdır. **Streamlit Community Cloud'da dosya sistemi kalıcı değildir**: uygulama
  yeniden başlatılınca ya da yeniden dağıtılınca kayıtlar silinir. Gerçek kullanımda barındırılan bir veritabanı
  (ör. PostgreSQL) gerekir; bu dosya aynı arayüzle ona taşınabilir.
* Tüm sorgular parametrelidir (SQL enjeksiyonu yok). Her sorgu şirket kimliğiyle sınırlanır; şirketsiz tek sorgu,
  müşteri ekranı için :meth:`Store.public_effective`'dir ve iç not ile kararı veren bilgisini ASLA döndürmez.
* Yazma sırasında kayıt :func:`casework.validate_record` ile yeniden doğrulanır (arayüz atlanmış olsa da).
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import casework as cw
import data_engine as de
import tenancy

__all__ = ["Store", "StoreError", "DB_ENV", "default_path", "open_store", "shared_store"]

DB_ENV = "ORBITAL_DB_PATH"
_COLUMNS = ("id", "company", "police_no", "seed", "action", "decision", "paid_tl", "pending_tl", "public_note",
            "internal_note", "decided_by", "decided_by_name", "decided_at", "auto_decision", "auto_paid_tl")
_DECISION_LIST = ",".join("'" + label.replace("'", "''") + "'" for label in de.DECISION_LABELS)
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  company TEXT NOT NULL,
  police_no TEXT NOT NULL,
  seed INTEGER NOT NULL,
  action TEXT NOT NULL CHECK (action IN ('karar','geri_al')),
  decision TEXT NOT NULL CHECK (action = 'geri_al' OR decision IN ({_DECISION_LIST})),
  paid_tl REAL NOT NULL CHECK (paid_tl >= 0),
  pending_tl REAL NOT NULL CHECK (pending_tl >= 0),
  public_note TEXT NOT NULL,
  internal_note TEXT NOT NULL,
  decided_by TEXT NOT NULL,
  decided_by_name TEXT NOT NULL,
  decided_at TEXT NOT NULL,
  auto_decision TEXT NOT NULL,
  auto_paid_tl REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_decisions_scope ON decisions (seed, company, police_no, id);
"""


class StoreError(RuntimeError):
    """Kayıt defterine erişilemedi (dosya yok/yazılamıyor/bozuk)."""


class Store:
    """SQLite karar defteri. Her işlemde yeni bağlantı açılır (Streamlit oturumları arasında güvenli)."""

    def __init__(self, path: str) -> None:
        """Dosyayı (ve üst klasörünü) oluşturur, tabloyu kurar.

        Args:
            path: SQLite dosya yolu.

        Raises:
            StoreError: Klasör ya da dosya oluşturulamıyorsa.
        """
        self.path = path
        try:
            folder = os.path.dirname(os.path.abspath(path))
            os.makedirs(folder, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
        except (OSError, sqlite3.Error) as exc:
            raise StoreError(f"Karar defteri açılamadı: {exc}") from exc

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row(r: sqlite3.Row) -> cw.Decision:
        return cw.Decision(
            id=int(r["id"]), company=str(r["company"]), police_no=str(r["police_no"]), seed=int(r["seed"]),
            action=str(r["action"]), decision=str(r["decision"]), paid_tl=float(r["paid_tl"]),
            pending_tl=float(r["pending_tl"]), public_note=str(r["public_note"]), internal_note=str(r["internal_note"]),
            decided_by=str(r["decided_by"]), decided_by_name=str(r["decided_by_name"]),
            decided_at=str(r["decided_at"]), auto_decision=str(r["auto_decision"]), auto_paid_tl=float(r["auto_paid_tl"]))

    def add(self, record: cw.Decision) -> int:
        """Kaydı doğrulayıp ekler.

        Args:
            record: Yeni kayıt (``id`` yok sayılır).

        Returns:
            Yeni kaydın numarası.

        Raises:
            CaseworkError: Kayıt geçersizse.
            StoreError: Yazılamazsa.
        """
        cw.validate_record(record)
        values = [getattr(record, c) for c in _COLUMNS[1:]]
        sql = f"INSERT INTO decisions ({', '.join(_COLUMNS[1:])}) VALUES ({', '.join('?' for _ in values)})"
        try:
            with self._connect() as conn:
                return int(conn.execute(sql, values).lastrowid or 0)
        except sqlite3.Error as exc:
            raise StoreError(f"Karar kaydedilemedi: {exc}") from exc

    def _select(self, where: str, args: Sequence[object], order: str, limit: Optional[int] = None) -> List[cw.Decision]:
        sql = f"SELECT {', '.join(_COLUMNS)} FROM decisions WHERE {where} ORDER BY id {order}"
        params = list(args)
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        try:
            with self._connect() as conn:
                return [self._row(r) for r in conn.execute(sql, params).fetchall()]
        except sqlite3.Error as exc:
            raise StoreError(f"Karar defteri okunamadı: {exc}") from exc

    def history(self, company: str, seed: int, police_no: Optional[str] = None, limit: int = 2000) -> List[cw.Decision]:
        """Bir şirketin kayıtlarını YENİDEN ESKİYE döndürür (iç not dahil; yalnızca sigortacı ekranı içindir).

        Args:
            company: Şirket kimliği (zorunlu; başka şirketin kaydı döndürülmez).
            seed: Veri tohumu.
            police_no: İsteğe bağlı tek poliçe.
            limit: En çok satır.

        Raises:
            CaseworkError: Şirket bilinmiyorsa.
        """
        if not tenancy.valid_company(company):
            raise cw.CaseworkError("Bilinmeyen şirket.")
        where, args = "company = ? AND seed = ?", [company, int(seed)]
        if police_no is not None:
            where += " AND police_no = ?"
            args.append(str(police_no))
        return self._select(where, args, "DESC", max(1, min(int(limit), 10_000)))

    def effective(self, company: str, seed: int) -> Dict[str, cw.Decision]:
        """Bir şirketin poliçelerinin geçerli kararları (iç not dahil; sigortacı ekranı içindir)."""
        if not tenancy.valid_company(company):
            raise cw.CaseworkError("Bilinmeyen şirket.")
        return cw.effective(self._select("company = ? AND seed = ?", [company, int(seed)], "ASC"))

    def public_effective(self, seed: int) -> Dict[str, cw.Decision]:
        """Tüm şirketlerin geçerli kararları, müşteri ekranı için: iç not ve kararı veren bilgisi BOŞALTILMIŞTIR."""
        full = cw.effective(self._select("seed = ?", [int(seed)], "ASC"))
        return {p: replace(d, internal_note="", decided_by="-", decided_by_name="-") for p, d in full.items()}


def default_path() -> str:
    """Dosya yolu: ``ORBITAL_DB_PATH`` ortam değişkeni, yoksa uygulama klasöründeki ``orbital_data/`` altı."""
    env = os.environ.get(DB_ENV, "").strip()
    return env or os.path.join(os.path.dirname(os.path.abspath(__file__)), "orbital_data", "orbital.sqlite3")


def open_store(path: Optional[str] = None) -> Store:
    """Defteri açar; varsayılan yol yazılamazsa geçici klasöre düşer (kalıcılık daha da zayıftır).

    Args:
        path: Açık yol; ``None`` ise :func:`default_path`.

    Returns:
        ``Store``.

    Raises:
        StoreError: Hiçbir yol kullanılamıyorsa. Açıkça verilen yol bozuksa geçici klasöre DÜŞÜLMEZ.
    """
    if path is not None:
        return Store(path)
    try:
        return Store(default_path())
    except StoreError:
        return Store(os.path.join(tempfile.gettempdir(), "orbital_field_decisions.sqlite3"))


#: Süreç boyunca paylaşılan defter (Streamlit her çalıştırmada ``app.py``'yi yeniden yürütür; bu modül yeniden yüklenmez).
_SHARED: Dict[str, Store] = {}


def shared_store() -> Store:
    """Varsayılan yoldaki defteri döndürür; aynı yol için tek örnek tutulur (her çalıştırmada şema yeniden kurulmaz).

    Returns:
        ``Store``.

    Raises:
        StoreError: Defter açılamazsa.
    """
    key = default_path()
    cached = _SHARED.get(key)
    if cached is not None and os.path.exists(cached.path):
        return cached
    opened = open_store()
    _SHARED[key] = opened
    return opened
