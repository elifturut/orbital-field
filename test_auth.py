"""auth.py çevrimdışı testleri (Streamlit gerekmez)."""
import hashlib
import json
import os
import sys
from dataclasses import FrozenInstanceError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auth  # noqa: E402
from auth import LoginState, Principal, Role  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f"  [{extra}]" if (extra and not cond) else ""))
    if not cond:
        FAILS.append(name)


ACC = auth.demo_accounts()
PW = auth.DEMO_PASSWORDS
T0 = 1_000_000.0

# --- demo hesapları
check("4 demo hesabı: 1 sigortacı + 3 müşteri", len(ACC) == 4 and sum(a.role is Role.INSURER for a in ACC) == 1)
check("kullanıcı adları benzersiz ve normal", len({a.username for a in ACC}) == 4 and all(a.username == auth.normalise_username(a.username) for a in ACC))
check("her demo parolası saklı özetle eşleşir", all(auth.verify_password(PW[a.username], a.salt_hex, a.hash_hex, a.iterations) for a in ACC))
check("demo parolaları birbirine uymaz", all(not auth.verify_password(PW[b.username], a.salt_hex, a.hash_hex, a.iterations) for a in ACC for b in ACC if a is not b))
check("demo tuzları benzersiz", len({a.salt_hex for a in ACC}) == 4)
check("demo özetleri benzersiz ve 32 bayt", len({a.hash_hex for a in ACC}) == 4 and all(len(bytes.fromhex(a.hash_hex)) == 32 for a in ACC))
check("demo tur sayısı yeterli", all(a.iterations >= auth.MIN_ITERATIONS for a in ACC))
check("demo parolaları düz metin olarak hesaplarda yok", all(PW[a.username] not in repr(a) for a in ACC))
check("müşteri sıralamaları 1,2,3", sorted(a.customer_rank for a in ACC if a.role is Role.CUSTOMER) == [1, 2, 3])
check("sigortacıda müşteri eşlemesi yok", all(a.customer_rank is None and a.customer_name is None for a in ACC if a.role is Role.INSURER))
try:
    ACC[0].username = "x"  # type: ignore[misc]
    check("hesap değiştirilemez", False)
except FrozenInstanceError:
    check("hesap değiştirilemez", True)

# --- parola özeti
s1, h1 = auth.hash_password("Parola-123", salt=b"\x01" * 16)
s2, h2 = auth.hash_password("Parola-123", salt=b"\x01" * 16)
s3, h3 = auth.hash_password("Parola-123", salt=b"\x02" * 16)
check("aynı tuz ve parola aynı özet", h1 == h2 and s1 == s2)
check("farklı tuz farklı özet", h1 != h3)
check("tuz verilmezse rastgele", auth.hash_password("x")[0] != auth.hash_password("x")[0])
check("özet = PBKDF2-HMAC-SHA256", h1 == hashlib.pbkdf2_hmac("sha256", b"Parola-123", b"\x01" * 16, auth.PBKDF2_ITERATIONS).hex())
check("doğru parola doğrulanır", auth.verify_password("Parola-123", s1, h1, auth.PBKDF2_ITERATIONS))
check("yanlış parola reddedilir", not auth.verify_password("parola-123", s1, h1, auth.PBKDF2_ITERATIONS))
check("farklı tur sayısı reddedilir", not auth.verify_password("Parola-123", s1, h1, auth.PBKDF2_ITERATIONS + 1))
check("bozuk onaltılık veri güvenle reddedilir", not auth.verify_password("x", "zz", "yy", 200000))
check("Türkçe karakterli parola", auth.verify_password("şifre-ÇĞİ-1", *auth.hash_password("şifre-ÇĞİ-1"), auth.PBKDF2_ITERATIONS))
for bad, label in ((lambda: auth.hash_password(""), "boş parola"), (lambda: auth.hash_password("x", iterations=10), "düşük tur")):
    try:
        bad()
        check(f"hata: {label}", False)
    except ValueError:
        check(f"hata: {label}", True)
check("kullanıcı adı normalleşir", auth.normalise_username("  SigortaCI ") == "sigortaci" and auth.normalise_username(None) == "")

# --- giriş
for acc in ACC:
    res = auth.authenticate(ACC, acc.username, PW[acc.username], LoginState(), T0)
    check(f"giriş başarılı: {acc.username}", res.code == "ok" and res.principal is not None and res.principal.role is acc.role
          and res.principal.username == acc.username and res.principal.customer_rank == acc.customer_rank)
check("kullanıcı adında boşluk/büyük harf", auth.authenticate(ACC, "  SIGORTACI ", PW["sigortaci"], LoginState(), T0).code == "ok")
check("parolada büyük/küçük harf önemli", auth.authenticate(ACC, "sigortaci", PW["sigortaci"].lower(), LoginState(), T0).code == "invalid")
check("parola başka hesaba ait", auth.authenticate(ACC, "musteri1", PW["musteri2"], LoginState(), T0).code == "invalid")
r_bad = auth.authenticate(ACC, "sigortaci", "yanlis", LoginState(), T0)
r_unknown = auth.authenticate(ACC, "yok-kullanici", "yanlis", LoginState(), T0)
check("yanlış parola ve olmayan kullanıcı aynı yanıtı verir", (r_bad.code, r_bad.principal, r_bad.retry_after) == (r_unknown.code, r_unknown.principal, r_unknown.retry_after) == ("invalid", None, 0.0))
check("boş alan: empty", auth.authenticate(ACC, "", "x", LoginState(), T0).code == "empty" and auth.authenticate(ACC, "sigortaci", "", LoginState(), T0).code == "empty")
check("tür dışı girdi güvenle reddedilir", all(auth.authenticate(ACC, u, p, LoginState(), T0).principal is None for u, p in ((None, None), (123, 456), ("sigortaci", None), ("sigortaci", 123), (["a"], ["b"]))))
check("çok uzun kullanıcı adı ve parola çökertmez", auth.authenticate(ACC, "a" * 5000, "b" * 5000, LoginState(), T0).code == "invalid" and auth.authenticate(ACC, "sigortaci", "b" * 5000, LoginState(), T0).code == "invalid")
check("çok uzun doğru-önekli parola kabul edilmez", auth.authenticate(ACC, "sigortaci", PW["sigortaci"] + "x" * 300, LoginState(), T0).principal is None)

# yanıt süresi sızıntısı yok: her durumda tam bir PBKDF2 hesabı
calls = []
real = hashlib.pbkdf2_hmac
auth.hashlib.pbkdf2_hmac = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
try:
    for u, p in (("sigortaci", "yanlis"), ("yok", "yanlis"), ("sigortaci", PW["sigortaci"]), ("a" * 100, "yanlis")):
        calls.clear()
        auth.authenticate(ACC, u, p, LoginState(), T0)
        check(f"tam bir PBKDF2 hesabı: {u[:9]}/{'doğru' if p == PW.get(u) else 'yanlış'}", len(calls) == 1, len(calls))
    calls.clear()
    auth.authenticate(ACC, "", "x", LoginState(), T0)
    check("boş girişte hesap yapılmaz", not calls)
finally:
    auth.hashlib.pbkdf2_hmac = real

# --- kilitleme
st = LoginState()
codes = [auth.authenticate(ACC, "sigortaci", "yanlis", st, T0 + i).code for i in range(auth.MAX_FAILURES)]
check("4 hata invalid, 5. hata locked", codes == ["invalid"] * (auth.MAX_FAILURES - 1) + ["locked"], codes)
check("kilit süresi ayarlandı", abs(st.locked_until - (T0 + auth.MAX_FAILURES - 1 + auth.LOCK_SECONDS)) < 1e-6 and st.failures == 0)
lock_now = T0 + auth.MAX_FAILURES
res = auth.authenticate(ACC, "sigortaci", PW["sigortaci"], st, lock_now)
check("kilitliyken doğru parola da reddedilir", res.code == "locked" and res.principal is None and 0 < res.retry_after <= auth.LOCK_SECONDS)
res = auth.authenticate(ACC, "sigortaci", PW["sigortaci"], st, lock_now + auth.LOCK_SECONDS - 5)
check("kilit bitmeden hâlâ reddedilir", res.code == "locked" and res.retry_after > 0)
res = auth.authenticate(ACC, "sigortaci", PW["sigortaci"], st, lock_now + auth.LOCK_SECONDS + 1)
check("kilit bitince doğru parola girer", res.code == "ok" and st.locked_until == 0.0 and st.failures == 0)
st = LoginState()
for i in range(auth.MAX_FAILURES - 1):
    auth.authenticate(ACC, "musteri1", "yanlis", st, T0 + i)
check("4 hata sonrası sayaç 4", st.failures == auth.MAX_FAILURES - 1)
auth.authenticate(ACC, "musteri1", PW["musteri1"], st, T0 + 10)
check("başarılı giriş sayacı sıfırlar", st.failures == 0 and st.locked_until == 0.0)
st = LoginState()
auth.authenticate(ACC, "", "", st, T0)
auth.authenticate(ACC, "musteri1", "", st, T0)
check("boş denemeler sayaca eklenmez", st.failures == 0)
st = LoginState()
for i in range(auth.MAX_FAILURES):
    auth.authenticate(ACC, f"yok{i}", "yanlis", st, T0 + i)
check("olmayan kullanıcılarla denemek de kilitler", st.locked_until > 0)
st1, st2 = LoginState(), LoginState()
for i in range(auth.MAX_FAILURES):
    auth.authenticate(ACC, "sigortaci", "yanlis", st1, T0 + i)
check("kilit yalnızca ilgili oturumun sayacında", st1.locked_until > 0 and st2.locked_until == 0 and auth.authenticate(ACC, "sigortaci", PW["sigortaci"], st2, T0).code == "ok")

# --- oturum süresi
check("taze oturum geçerli", not auth.session_expired(T0, T0 + 10))
check("tam süre sınırında hâlâ geçerli", not auth.session_expired(T0, T0 + auth.IDLE_TIMEOUT_S))
check("süre aşılınca bitmiş", auth.session_expired(T0, T0 + auth.IDLE_TIMEOUT_S + 1))
check("zaman damgası yok/bozuk: bitmiş", all(auth.session_expired(v, T0) for v in (None, "x", True, [1], {})))
check("gelecekteki zaman damgası şüpheli: bitmiş", auth.session_expired(T0 + 1000, T0))
check("özel zaman aşımı", auth.session_expired(T0, T0 + 61, timeout=60) and not auth.session_expired(T0, T0 + 59, timeout=60))

# --- yetkilendirme
ins = Principal("sigortaci", Role.INSURER, "S")
cus = Principal("musteri1", Role.CUSTOMER, "M", customer_rank=1)
all_perms = {auth.PERM_PORTAL_ANY, auth.PERM_INSURANCE, auth.PERM_EVIDENCE, auth.PERM_CALIBRATION, auth.PERM_OVERVIEW, auth.PERM_SECTORS}
check("sigortacı tüm iç izinlere sahip", all(auth.allowed(ins, p) for p in all_perms))
check("sigortacı müşteri görünümünü önizleyebilir", auth.allowed(ins, auth.PERM_PORTAL_ANY))
check("müşteri yalnızca kendi portalını görür", auth.permissions_for(cus) == frozenset({auth.PERM_PORTAL_OWN}))
check("müşteri hiçbir iç izne sahip değil", not any(auth.allowed(cus, p) for p in all_perms))
check("oturumsuz kimse izin alamaz", auth.permissions_for(None) == frozenset() and not auth.allowed(None, auth.PERM_PORTAL_OWN))
for perm in sorted(all_perms):
    try:
        auth.require(cus, perm)
        check(f"müşteri require reddedilir: {perm}", False)
    except PermissionError as exc:
        check(f"müşteri require reddedilir: {perm}", "Müşteri" in str(exc))
try:
    auth.require(None, auth.PERM_PORTAL_OWN)
    check("oturumsuz require reddedilir", False)
except PermissionError as exc:
    check("oturumsuz require reddedilir", "açılmamış" in str(exc))
try:
    auth.require(ins, auth.PERM_INSURANCE)
    auth.require(cus, auth.PERM_PORTAL_OWN)
    check("izinli require sessiz geçer", True)
except PermissionError:
    check("izinli require sessiz geçer", False)
check("bilinmeyen izin hiçbir role verilmez", not auth.allowed(ins, "yok.izin") and not auth.allowed(cus, "yok.izin"))
check("sekmeler: sigortacı yedi, müşteri/oturumsuz sıfır", auth.tabs_for(ins) == auth.INSURER_TABS and len(auth.INSURER_TABS) == 7 and auth.tabs_for(cus) == () and auth.tabs_for(None) == ())
check("müşteri portalı sekmesi sigortacıda var", "MÜŞTERİ PORTALI" in auth.INSURER_TABS)

# --- oturum serileştirme
check("oturum gidiş-dönüş (sigortacı)", Principal.from_session(ins.to_session()) == ins)
check("oturum gidiş-dönüş (müşteri sıra)", Principal.from_session(cus.to_session()) == cus)
named = Principal("m", Role.CUSTOMER, "M", customer_name="H. Kaya")
check("oturum gidiş-dönüş (müşteri ad)", Principal.from_session(named.to_session()) == named)
bad_sessions = {
    "dict değil": "x", "None": None, "rol yok": {"username": "a"}, "bilinmeyen rol": {"username": "a", "role": "yonetici"},
    "kullanıcı yok": {"role": "sigortaci"},
    "müşteri eşlemesiz": {"username": "a", "role": "musteri"},
    "müşteri iki eşlemeli": {"username": "a", "role": "musteri", "customer_rank": 1, "customer_name": "H. Kaya"},
    "sigortacıda sıra": {"username": "a", "role": "sigortaci", "customer_rank": 1},
    "sigortacıda ad": {"username": "a", "role": "sigortaci", "customer_name": "H. Kaya"},
    "sıra 0": {"username": "a", "role": "musteri", "customer_rank": 0},
    "sıra negatif": {"username": "a", "role": "musteri", "customer_rank": -1},
    "sıra bool": {"username": "a", "role": "musteri", "customer_rank": True},
    "sıra metin": {"username": "a", "role": "musteri", "customer_rank": "1"},
}
for label, data in bad_sessions.items():
    check(f"bozuk oturum reddedilir: {label}", Principal.from_session(data) is None)

# --- gerçek hesaplar (JSON)
def entry(user, role, pw, **extra):
    salt, digest = auth.hash_password(pw)
    return {"username": user, "role": role, "display_name": user.title(), "salt": salt, "hash": digest, **extra}


good = json.dumps([entry("Ayse.Demir", "musteri", "Gizli-Parola-1", customer_name="A. Demir"), entry("uzman", "sigortaci", "Baska-Parola-2")])
real = auth.parse_accounts(good)
check("JSON: iki hesap, kullanıcı adı normalleşir", [a.username for a in real] == ["ayse.demir", "uzman"] and real[0].customer_name == "A. Demir" and real[0].customer_rank is None)
check("JSON hesabıyla giriş", auth.authenticate(real, "AYSE.demir", "Gizli-Parola-1", LoginState(), T0).code == "ok"
      and auth.authenticate(real, "AYSE.demir", "Gizli-Parola-2", LoginState(), T0).code == "invalid")
check("JSON hesapları varken demo parolası geçmez", auth.authenticate(real, "sigortaci", PW["sigortaci"], LoginState(), T0).code == "invalid")
check("JSON müşteri oturumu ad eşlemeli", auth.authenticate(real, "ayse.demir", "Gizli-Parola-1", LoginState(), T0).principal.customer_name == "A. Demir")
bad_json = {
    "geçersiz JSON": "{",
    "liste değil": "{}",
    "boş liste": "[]",
    "nesne değil": "[1]",
    "kullanıcı adı yok": json.dumps([{**entry("a", "sigortaci", "p"), "username": ""}]),
    "yinelenen ad": json.dumps([entry("a", "sigortaci", "p"), entry("A", "sigortaci", "q")]),
    "bilinmeyen rol": json.dumps([entry("a", "yonetici", "p")]),
    "onaltılık değil": json.dumps([{**entry("a", "sigortaci", "p"), "hash": "xyz"}]),
    "kısa özet": json.dumps([{**entry("a", "sigortaci", "p"), "hash": "ab" * 8}]),
    "kısa tuz": json.dumps([{**entry("a", "sigortaci", "p"), "salt": "ab" * 4}]),
    "düşük tur": json.dumps([entry("a", "sigortaci", "p", iterations=1000)]),
    "tur bool": json.dumps([entry("a", "sigortaci", "p", iterations=True)]),
    "müşteri adı yok": json.dumps([entry("a", "musteri", "p")]),
    "müşteri adı boş": json.dumps([entry("a", "musteri", "p", customer_name="  ")]),
    "sigortacıda müşteri adı": json.dumps([entry("a", "sigortaci", "p", customer_name="X")]),
    "çok uzun kullanıcı adı": json.dumps([entry("a" * 65, "sigortaci", "p")]),
}
for label, text in bad_json.items():
    try:
        auth.parse_accounts(text)
        check(f"JSON reddedilir: {label}", False)
    except ValueError:
        check(f"JSON reddedilir: {label}", True)

# hesap kaydı aracı: üretilen kayıt parse_accounts ve authenticate ile uçtan uca çalışır
e_ins = auth.account_entry("  Sirket ", auth.Role.INSURER, "Şirket A.Ş.", "Uzun-Parola-12!")
e_cus = auth.account_entry("Ayse", auth.Role.CUSTOMER, "Ayşe Y.", "Baska-Parola-34!", " A. Yılmaz ")
check("hesap kaydı: kullanıcı adı küçük harf ve kırpılmış", e_ins["username"] == "sirket" and e_cus["username"] == "ayse")
check("hesap kaydı: düz parola yok, tur sayısı varsayılan", "Uzun-Parola-12!" not in json.dumps(e_ins) and e_ins["iterations"] == auth.PBKDF2_ITERATIONS and "password" not in e_ins)
check("hesap kaydı: müşteri adı kırpılır, sigortacıda yok", e_cus["customer_name"] == "A. Yılmaz" and "customer_name" not in e_ins)
made = auth.parse_accounts(json.dumps([e_ins, e_cus]))
check("hesap kaydı: üretilen JSON ayrıştırılır", [a.username for a in made] == ["sirket", "ayse"] and made[1].customer_name == "A. Yılmaz")
check("hesap kaydı: parola ile giriş çalışır", auth.authenticate(made, "SIRKET", "Uzun-Parola-12!", LoginState(), T0).code == "ok"
      and auth.authenticate(made, "ayse", "Baska-Parola-34!", LoginState(), T0).principal.customer_name == "A. Yılmaz")
check("hesap kaydı: yanlış parola geçmez", auth.authenticate(made, "ayse", "Uzun-Parola-12!", LoginState(), T0).code == "invalid")
check("hesap kaydı: her çağrıda farklı tuz", auth.account_entry("a", auth.Role.INSURER, "", "x")["salt"] != auth.account_entry("a", auth.Role.INSURER, "", "x")["salt"])
for label, args in {"boş kullanıcı adı": ("", auth.Role.INSURER, "x", "p", None), "boş parola": ("a", auth.Role.INSURER, "x", "", None),
                    "çok uzun parola": ("a", auth.Role.INSURER, "x", "p" * 257, None), "müşteri adı yok": ("a", auth.Role.CUSTOMER, "x", "p", None),
                    "müşteri adı boşluk": ("a", auth.Role.CUSTOMER, "x", "p", "  "), "sigortacıda müşteri adı": ("a", auth.Role.INSURER, "x", "p", "X")}.items():
    try:
        auth.account_entry(*args)
        check(f"hesap kaydı reddedilir: {label}", False)
    except ValueError:
        check(f"hesap kaydı reddedilir: {label}", True)

print()
print("BAŞARISIZ: " + ", ".join(FAILS) if FAILS else "BAŞARISIZ: yok — tüm kontroller geçti")
sys.exit(1 if FAILS else 0)
