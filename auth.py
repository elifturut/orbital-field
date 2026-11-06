"""Kimlik doğrulama ve yetkilendirme çekirdeği (saf Python; Streamlit içe aktarmaz).

İki rol vardır: **Sigortacı** (kendi şirketinin poliçeleri, müşteri dosyaları ve karar verme) ve **Müşteri** (yalnızca kendi
poliçeleri ve hasar durumu). Bu modül yalnızca KARAR verir; hangi verinin gösterileceğini ``portal.py``,
ekrana basmayı ``app.py`` yapar.

Güvenlik kapsamı (dürüst sınır)
-------------------------------
* Parolalar PBKDF2-HMAC-SHA256 ile (kullanıcıya özel tuz, 200 bin tur) saklanır; düz metin tutulmaz.
* Karşılaştırma sabit zamanlıdır; olmayan kullanıcı için de aynı hesap yapılır (kullanıcı adı sızdırmaz).
* Hatalı denemelerde oturum başına kilitlenir. Bu **oturum düzeyindedir**: yeni oturum açan saldırgan sayacı
  sıfırlar. Gerçek kullanımda hız sınırı, kimlik sağlayıcı (OIDC/SSO) ve çok adımlı doğrulama gerekir.
* Dahili demo hesapları herkese açık sentetik veri içindir. Gerçek hesaplar ``ORBITAL_ACCOUNTS_JSON``
  ortam değişkeniyle verilir (bkz. :func:`parse_accounts`); o zaman demo hesapları yüklenmez.
* Yetki denetimi sunucu tarafındadır: oturum durumu tarayıcıdan değiştirilemez.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import tenancy

__all__ = [
    "Role", "Account", "Principal", "LoginState", "AuthResult", "ROLE_LABELS", "DEMO_ACCOUNTS", "DEMO_PASSWORDS",
    "PBKDF2_ITERATIONS", "MAX_FAILURES", "LOCK_SECONDS", "IDLE_TIMEOUT_S", "MIN_ITERATIONS",
    "hash_password", "verify_password", "normalise_username", "demo_accounts", "parse_accounts", "authenticate",
    "session_expired", "permissions_for", "allowed", "require", "tabs_for", "INSURER_TABS", "account_entry",
    "PERM_DECIDE", "PERM_ADMIN", "MAX_TITLE",
]

PBKDF2_ITERATIONS = 200_000
MIN_ITERATIONS = 100_000
MAX_FAILURES = 5
LOCK_SECONDS = 60.0
IDLE_TIMEOUT_S = 30 * 60.0
MAX_USERNAME = 64
MAX_PASSWORD = 256
MAX_TITLE = 80


class Role(str, Enum):
    """Kullanıcı rolü."""

    INSURER = "sigortaci"
    CUSTOMER = "musteri"


ROLE_LABELS: Dict[Role, str] = {Role.INSURER: "Sigortacı", Role.CUSTOMER: "Müşteri"}

#: İzin adları: uygulamadaki her korumalı ekran/işlem bir izne bağlıdır.
PERM_PORTAL_OWN = "portal.own"            # kendi poliçe ve hasar durumu
PERM_PORTAL_ANY = "portal.any"            # herhangi bir müşterinin görünümünü önizleme
PERM_INSURANCE = "insurance.dashboard"    # tüm poliçeler, risk skoru, kural paneli
PERM_EVIDENCE = "insurance.evidence"      # karar kanıt paketi (iç ölçümler, eşikler)
PERM_CALIBRATION = "insurance.calibration"
PERM_OVERVIEW = "platform.overview"
PERM_SECTORS = "platform.sectors"         # tarım, belediye, kurtarma
PERM_DECIDE = "casework.decide"           # müşteri dosyası açma ve tazminat kararı verme/geri alma
PERM_ADMIN = "company.admin"              # şirket profili, ekip, karar kaydı (yalnızca kendi şirketi)

ROLE_PERMISSIONS: Dict[Role, FrozenSet[str]] = {
    Role.INSURER: frozenset({PERM_PORTAL_ANY, PERM_INSURANCE, PERM_EVIDENCE, PERM_CALIBRATION, PERM_OVERVIEW,
                             PERM_SECTORS, PERM_DECIDE, PERM_ADMIN}),
    Role.CUSTOMER: frozenset({PERM_PORTAL_OWN}),
}

#: Sigortacı için ana sekme sırası: dört çekirdek sekme ve tarım/belediye/kurtarmayı toplayan genişleme sekmesi.
INSURER_TABS: Tuple[str, ...] = ("GENEL BAKIŞ", "SİGORTA", "MÜŞTERİ DOSYALARI", "KALİBRASYON", "DİĞER SEKTÖRLER",
                                 "YÖNETİM")


@dataclass(frozen=True)
class Account:
    """Kayıtlı bir hesap (parola düz metin değil, özet olarak tutulur).

    Attributes:
        username: Normalleştirilmiş kullanıcı adı.
        role: Rol.
        display_name: Ekranda görünen ad.
        salt_hex: Tuz (onaltılık).
        hash_hex: PBKDF2 özeti (onaltılık).
        iterations: PBKDF2 tur sayısı.
        customer_rank: Müşteri hesabı için: ``portal.customer_ranking`` sıralamasında kaçıncı müşteri (1 = en çeşitli
            kararlara sahip olan). Yalnızca demo verisinde kullanılır.
        customer_name: Müşteri hesabı için: ``sigortali`` adı (gerçek eşleme). ``customer_rank`` ile birlikte verilemez.
        company: Sigortacı hesabı için bağlı olduğu şirketin kimliği (``tenancy``); müşteride ``None``.
        title: Sigortacının unvanı (ör. "Hasar yöneticisi"); yalnızca görüntü amaçlıdır, yetki vermez.
    """

    username: str
    role: Role
    display_name: str
    salt_hex: str
    hash_hex: str
    iterations: int = PBKDF2_ITERATIONS
    customer_rank: Optional[int] = None
    customer_name: Optional[str] = None
    company: Optional[str] = None
    title: str = ""


@dataclass(frozen=True)
class Principal:
    """Oturum açmış kullanıcı (parola özetini taşımaz).

    Attributes:
        username: Kullanıcı adı.
        role: Rol.
        display_name: Ekranda görünen ad.
        customer_rank: Müşteri için sıralama eşlemesi (demo).
        customer_name: Müşteri için ad eşlemesi (gerçek).
        company: Sigortacının şirket kimliği (``tenancy``); sigortacı verisi bu kimlikle süzülür. Müşteride ``None``.
        title: Sigortacının unvanı (görüntü).
    """

    username: str
    role: Role
    display_name: str
    customer_rank: Optional[int] = None
    customer_name: Optional[str] = None
    company: Optional[str] = None
    title: str = ""

    def to_session(self) -> Dict[str, Any]:
        """Oturum durumuna yazılacak düz sözlük."""
        return {"username": self.username, "role": self.role.value, "display_name": self.display_name,
                "customer_rank": self.customer_rank, "customer_name": self.customer_name,
                "company": self.company, "title": self.title}

    @staticmethod
    def from_session(data: Any) -> Optional["Principal"]:
        """Oturum durumundan ``Principal`` kurar; bozuk ya da yabancı veri ``None`` verir."""
        if not isinstance(data, Mapping):
            return None
        try:
            role = Role(str(data.get("role")))
            username = str(data["username"])
            name = str(data.get("display_name", username))
        except (KeyError, ValueError):
            return None
        rank, cname = data.get("customer_rank"), data.get("customer_name")
        if rank is not None and (not isinstance(rank, int) or isinstance(rank, bool) or rank < 1):
            return None
        if role is Role.CUSTOMER and (rank is None) == (cname is None):
            return None                       # müşterinin tam olarak bir eşleme anahtarı olmalı
        if role is Role.INSURER and (rank is not None or cname is not None):
            return None
        company, title = data.get("company"), data.get("title", "")
        if role is Role.INSURER and not tenancy.valid_company(company):
            return None                       # sigortacı geçerli bir şirkete bağlı olmalı (bozuk kayıt veri açmaz)
        if role is Role.CUSTOMER and company is not None:
            return None
        if not isinstance(title, str) or len(title) > MAX_TITLE:
            return None
        return Principal(username, role, name, rank, None if cname is None else str(cname), company, title)


@dataclass
class LoginState:
    """Oturum başına giriş denemesi sayacı (değiştirilebilir).

    Attributes:
        failures: Ardışık hatalı deneme sayısı.
        locked_until: Kilidin açılacağı an (saniye; ``0`` = kilitli değil).
    """

    failures: int = 0
    locked_until: float = 0.0


@dataclass(frozen=True)
class AuthResult:
    """Giriş denemesinin sonucu.

    Attributes:
        principal: Başarılıysa kullanıcı, değilse ``None``.
        code: ``ok`` | ``invalid`` | ``empty`` | ``locked``.
        retry_after: ``locked`` ise kalan saniye.
    """

    principal: Optional[Principal]
    code: str
    retry_after: float = 0.0


# --------------------------------------------------------------------------- #
# Parola özeti
# --------------------------------------------------------------------------- #
def hash_password(password: str, salt: Optional[bytes] = None, iterations: int = PBKDF2_ITERATIONS) -> Tuple[str, str]:
    """Parolanın PBKDF2-HMAC-SHA256 özetini üretir.

    Args:
        password: Düz parola.
        salt: Tuz; ``None`` ise ``os.urandom`` ile 16 bayt.
        iterations: Tur sayısı (en az :data:`MIN_ITERATIONS`).

    Returns:
        ``(tuz_hex, özet_hex)``.

    Raises:
        ValueError: Parola boşsa ya da tur sayısı çok düşükse.
    """
    if not password:
        raise ValueError("Parola boş olamaz.")
    if iterations < MIN_ITERATIONS:
        raise ValueError(f"Tur sayısı en az {MIN_ITERATIONS} olmalı.")
    raw_salt = os.urandom(16) if salt is None else salt
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), raw_salt, iterations)
    return raw_salt.hex(), digest.hex()


def verify_password(password: str, salt_hex: str, hash_hex: str, iterations: int) -> bool:
    """Parolayı saklı özetle sabit zamanlı karşılaştırır."""
    try:
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except ValueError:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, max(int(iterations), 1))
    return hmac.compare_digest(actual, expected)


def normalise_username(value: Any) -> str:
    """Kullanıcı adını boşluk ve büyük/küçük harf farkından arındırır."""
    return str(value or "").strip().casefold()


# --------------------------------------------------------------------------- #
# Hesaplar
# --------------------------------------------------------------------------- #
#: Demo parolaları (yalnızca herkese açık sentetik demo içindir; arayüzde gösterilir).
DEMO_PASSWORDS: Dict[str, str] = {
    "sigortaci": "Sigortaci-Demo-2026", "musteri1": "Musteri1-Demo-2026",
    "musteri2": "Musteri2-Demo-2026", "musteri3": "Musteri3-Demo-2026",
    "sigortaci2": "Sigortaci2-Demo-2026",
}

DEMO_ACCOUNTS: Tuple[Account, ...] = (
    Account("sigortaci", Role.INSURER, "Hasar yöneticisi (demo)", "925054f508db1831faf6962255902ecc",
            "8a153a09e9117a2a02adb4c1ba5eb41572b4bb7f3490ea3172895b21f7f9c058", company="boztepe",
            title="Hasar yöneticisi"),
    Account("sigortaci2", Role.INSURER, "Ekspertiz uzmanı (demo)", "3c9d1e7a54b2086f91d3a7c05e4b1288",
            "96a29993b6e799e69019693a7f143416719de4410ca7a8b8d2b5df0524dfa163", company="yaylacik",
            title="Ekspertiz uzmanı"),
    Account("musteri1", Role.CUSTOMER, "Müşteri 1 (demo)", "5c746e8237cb91fc33120a07154067d1",
            "e5e53464d20bc49958ca75dc349f4dfabcab8fbe9401467dec834b495253a328", customer_rank=1),
    Account("musteri2", Role.CUSTOMER, "Müşteri 2 (demo)", "0aea27192d4121a4cb420dc21dc721bb",
            "fe566b4ed53a06b996c091e00f3a095099237f5f9dc0fe48562e0afb05050a50", customer_rank=2),
    Account("musteri3", Role.CUSTOMER, "Müşteri 3 (demo)", "7e9617e784dd1b95c0003e83c70a02c4",
            "6a2b501d2c73550a641f7ec08d4c346892208c43fa66b85b675ef03ed82d1936", customer_rank=3),
)


def demo_accounts() -> Tuple[Account, ...]:
    """Dahili demo hesaplarını döndürür."""
    return DEMO_ACCOUNTS


def parse_accounts(text: str) -> Tuple[Account, ...]:
    """Ortam değişkenindeki JSON'dan gerçek hesapları okur.

    Biçim: ``[{"username": "...", "role": "sigortaci"|"musteri", "display_name": "...", "salt": "<hex>",
    "hash": "<hex>", "iterations": 200000, "customer_name": "<sigortali adı>", "company": "<şirket kimliği>",
    "title": "<unvan>"}]``. Özetler :func:`hash_password` ile üretilir. Müşteri hesabında ``customer_name``
    zorunludur. Sigortacı hesabında ``company`` (bkz. ``tenancy.COMPANIES``) bağlı olduğu şirketi seçer; verilmezse
    ``tenancy.DEFAULT_COMPANY`` kullanılır (tüm şirketleri görmek yerine tek şirkete kısıtlanır). Müşteri
    hesabında ``company`` olmamalıdır (müşteri tüm şirketlerdeki kendi poliçelerini görür).

    Args:
        text: JSON metni.

    Returns:
        Hesap demeti.

    Raises:
        ValueError: Biçim hatalıysa, kullanıcı adı yineleniyorsa, tur sayısı düşükse ya da hiç hesap yoksa.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Hesap JSON'u okunamadı: {exc.msg}") from exc
    if not isinstance(data, list) or not data:
        raise ValueError("Hesap listesi boş ya da liste değil.")
    accounts: List[Account] = []
    seen = set()
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Hesap {i + 1}: nesne bekleniyordu.")
        username = normalise_username(item.get("username"))
        if not username or len(username) > MAX_USERNAME:
            raise ValueError(f"Hesap {i + 1}: geçersiz kullanıcı adı.")
        if username in seen:
            raise ValueError(f"Hesap {i + 1}: kullanıcı adı yineleniyor ({username}).")
        seen.add(username)
        try:
            role = Role(str(item.get("role")))
        except ValueError as exc:
            raise ValueError(f"Hesap {i + 1}: rol 'sigortaci' ya da 'musteri' olmalı.") from exc
        salt, digest = str(item.get("salt", "")), str(item.get("hash", ""))
        try:
            bytes.fromhex(salt), bytes.fromhex(digest)
        except ValueError as exc:
            raise ValueError(f"Hesap {i + 1}: tuz ve özet onaltılık olmalı.") from exc
        if len(bytes.fromhex(salt)) < 8 or len(bytes.fromhex(digest)) != 32:
            raise ValueError(f"Hesap {i + 1}: tuz en az 8 bayt, özet 32 bayt olmalı.")
        iterations = item.get("iterations", PBKDF2_ITERATIONS)
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < MIN_ITERATIONS:
            raise ValueError(f"Hesap {i + 1}: tur sayısı en az {MIN_ITERATIONS} olmalı.")
        cname = item.get("customer_name")
        if role is Role.CUSTOMER and not (isinstance(cname, str) and cname.strip()):
            raise ValueError(f"Hesap {i + 1}: müşteri hesabında customer_name zorunlu.")
        if role is Role.INSURER and cname is not None:
            raise ValueError(f"Hesap {i + 1}: sigortacı hesabında customer_name olmamalı.")
        company, title = item.get("company"), item.get("title", "")
        if role is Role.CUSTOMER and company is not None:
            raise ValueError(f"Hesap {i + 1}: müşteri hesabında company olmamalı (müşteri tüm şirketleri görür).")
        if role is Role.INSURER:
            company = tenancy.DEFAULT_COMPANY if company is None else company
            if not tenancy.valid_company(company):
                known = ", ".join(tenancy.company_ids())
                raise ValueError(f"Hesap {i + 1}: bilinmeyen şirket (geçerli: {known}).")
        if not isinstance(title, str) or len(title.strip()) > MAX_TITLE:
            raise ValueError(f"Hesap {i + 1}: unvan metin olmalı (en çok {MAX_TITLE} karakter).")
        accounts.append(Account(username, role, str(item.get("display_name") or username), salt, digest, iterations,
                                None, cname.strip() if role is Role.CUSTOMER else None,
                                company if role is Role.INSURER else None, title.strip()))
    return tuple(accounts)


# --------------------------------------------------------------------------- #
# Giriş
# --------------------------------------------------------------------------- #
_DUMMY_SALT = "00" * 16
_DUMMY_HASH = "00" * 32


def authenticate(accounts: Sequence[Account], username: Any, password: Any, state: LoginState, now: float) -> AuthResult:
    """Giriş denemesini değerlendirir ve ``state`` sayacını günceller.

    * Kilitliyken doğru parola bile reddedilir.
    * Olmayan kullanıcı için de bir PBKDF2 hesabı yapılır; yanıt süresi kullanıcı varlığını ele vermez.
    * Hata iletisi hep aynıdır (``invalid``); hangi alanın yanlış olduğu söylenmez.

    Args:
        accounts: Geçerli hesaplar.
        username: Girilen kullanıcı adı.
        password: Girilen parola.
        state: Oturumun deneme sayacı (yerinde değiştirilir).
        now: Şimdiki an (saniye; test için dışarıdan verilir).

    Returns:
        ``AuthResult``.
    """
    if state.locked_until > now:
        return AuthResult(None, "locked", state.locked_until - now)
    if state.locked_until:
        state.locked_until, state.failures = 0.0, 0
    uname = normalise_username(username)
    pwd = password if isinstance(password, str) else ""
    if not uname or not pwd:
        return AuthResult(None, "empty")
    account = next((a for a in accounts if a.username == uname), None) if len(uname) <= MAX_USERNAME else None
    if account is not None and len(pwd) <= MAX_PASSWORD:
        good = verify_password(pwd, account.salt_hex, account.hash_hex, account.iterations)
    else:
        verify_password(pwd[:MAX_PASSWORD], _DUMMY_SALT, _DUMMY_HASH, PBKDF2_ITERATIONS)
        good = False
    if good and account is not None:
        state.failures, state.locked_until = 0, 0.0
        return AuthResult(Principal(account.username, account.role, account.display_name, account.customer_rank,
                                    account.customer_name, account.company, account.title), "ok")
    state.failures += 1
    if state.failures >= MAX_FAILURES:
        state.failures, state.locked_until = 0, now + LOCK_SECONDS
        return AuthResult(None, "locked", LOCK_SECONDS)
    return AuthResult(None, "invalid")


def session_expired(last_seen: Any, now: float, timeout: float = IDLE_TIMEOUT_S) -> bool:
    """Oturum boşta kalma süresini aştı mı? Okunamayan zaman damgası süresi dolmuş sayılır."""
    if isinstance(last_seen, bool) or not isinstance(last_seen, (int, float)):
        return True
    return (now - float(last_seen)) > timeout or float(last_seen) > now + 1.0


# --------------------------------------------------------------------------- #
# Yetkilendirme
# --------------------------------------------------------------------------- #
def permissions_for(principal: Optional[Principal]) -> FrozenSet[str]:
    """Kullanıcının izinleri; oturum yoksa boş küme."""
    return ROLE_PERMISSIONS.get(principal.role, frozenset()) if principal else frozenset()


def allowed(principal: Optional[Principal], permission: str) -> bool:
    """Kullanıcı bu izne sahip mi?"""
    return permission in permissions_for(principal)


def require(principal: Optional[Principal], permission: str) -> None:
    """İzin yoksa ``PermissionError`` fırlatır (savunma amaçlı ikinci kapı).

    Raises:
        PermissionError: Oturum yoksa ya da rol bu izne sahip değilse.
    """
    if principal is None:
        raise PermissionError("Oturum açılmamış.")
    if permission not in permissions_for(principal):
        raise PermissionError(f"{ROLE_LABELS[principal.role]} rolü bu bölüme erişemez.")


def tabs_for(principal: Optional[Principal]) -> Tuple[str, ...]:
    """Rolün görebileceği sekmeler; müşteri tek sayfalı portal gördüğü için sekme listesi boştur."""
    if principal is not None and principal.role is Role.INSURER:
        return INSURER_TABS
    return ()



# --------------------------------------------------------------------------- #
# Hesap kaydı üretimi (yönetici aracı)
# --------------------------------------------------------------------------- #
def account_entry(username: str, role: Role, display_name: str, password: str,
                  customer_name: Optional[str] = None, company: Optional[str] = None,
                  title: str = "") -> Dict[str, Any]:
    """``ORBITAL_ACCOUNTS_JSON`` listesine eklenecek tek hesap kaydını üretir (parola özetli, düz metinsiz).

    Args:
        username: Kullanıcı adı (küçük harfe çevrilir).
        role: Rol.
        display_name: Ekranda görünen ad.
        password: Düz parola (kayda yazılmaz; yalnızca özeti yazılır).
        customer_name: Müşteri için ``sigortali`` adı (birebir eşleşmelidir); sigortacıda ``None``.
        company: Sigortacı için şirket kimliği (``tenancy``); müşteride ``None``. Sigortacıda verilmezse varsayılan şirket.
        title: Sigortacı unvanı (isteğe bağlı).

    Returns:
        JSON'a çevrilebilir sözlük.

    Raises:
        ValueError: Kullanıcı adı ya da parola boş/çok uzunsa, ya da rol ile ``customer_name`` uyuşmuyorsa.
    """
    uname = normalise_username(username)
    if not uname or len(uname) > MAX_USERNAME:
        raise ValueError("Geçersiz kullanıcı adı.")
    if not password or len(password) > MAX_PASSWORD:
        raise ValueError("Parola boş ya da çok uzun.")
    if role is Role.CUSTOMER and not (customer_name and customer_name.strip()):
        raise ValueError("Müşteri hesabında müşteri adı (sigortalı) zorunlu.")
    if role is Role.INSURER and customer_name:
        raise ValueError("Sigortacı hesabında müşteri adı olmamalı.")
    if role is Role.CUSTOMER and company:
        raise ValueError("Müşteri hesabında şirket olmamalı (müşteri tüm şirketleri görür).")
    if role is Role.INSURER:
        company = company or tenancy.DEFAULT_COMPANY
        if not tenancy.valid_company(company):
            raise ValueError(f"Bilinmeyen şirket (geçerli: {', '.join(tenancy.company_ids())}).")
    if len(title.strip()) > MAX_TITLE:
        raise ValueError(f"Unvan en çok {MAX_TITLE} karakter olabilir.")
    salt, digest = hash_password(password)
    entry: Dict[str, Any] = {"username": uname, "role": role.value, "display_name": display_name or uname,
                             "salt": salt, "hash": digest, "iterations": PBKDF2_ITERATIONS}
    if role is Role.CUSTOMER and customer_name:
        entry["customer_name"] = customer_name.strip()
    if role is Role.INSURER:
        entry["company"] = company
        if title.strip():
            entry["title"] = title.strip()
    return entry


def _main() -> None:  # pragma: no cover — etkileşimli yönetici aracı
    """``python auth.py`` : hesap kaydı üretir. Parola ekranda görünmez ve komut satırına yazılmaz."""
    import getpass

    username = input("Kullanıcı adı: ")
    role = Role.CUSTOMER if input("Rol [sigortaci/musteri]: ").strip().lower() == "musteri" else Role.INSURER
    name = input("Görünen ad: ")
    customer = input("Müşteri adı (sigortalı, birebir): ") if role is Role.CUSTOMER else None
    company = title = None
    if role is Role.INSURER:
        company = input(f"Şirket kimliği [{', '.join(tenancy.company_ids())}]: ").strip() or None
        title = input("Unvan (isteğe bağlı): ")
    entry = account_entry(username, role, name, getpass.getpass("Parola: "), customer, company, title or "")
    print(json.dumps(entry, ensure_ascii=False, indent=2))
    print("\nBu kaydı ORBITAL_ACCOUNTS_JSON listesine ekleyin (köşeli parantezli JSON listesi).")


if __name__ == "__main__":  # pragma: no cover
    _main()
