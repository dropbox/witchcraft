class Pickbrain < Formula
  desc "Semantic search over Claude Code and Codex conversations"
  homepage "https://github.com/dropbox/witchcraft"
  version "0.1.0"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/dropbox/witchcraft/releases/download/v#{version}/pickbrain-#{version}-aarch64-apple-darwin.tar.gz"
      sha256 "PLACEHOLDER"
    end
    on_intel do
      url "https://github.com/dropbox/witchcraft/releases/download/v#{version}/pickbrain-#{version}-x86_64-apple-darwin.tar.gz"
      sha256 "PLACEHOLDER"
    end
  end

  def install
    bin.install "pickbrain"
    (share/"pickbrain/skills/pickbrain").install "skills/pickbrain/SKILL.md"
    (share/"pickbrain/skills/pickbrain-codex").install "skills/pickbrain-codex/SKILL.md"
  end

  def post_install
    claude_dir = Pathname.new(Dir.home)/".claude/skills/pickbrain"
    codex_dir = Pathname.new(Dir.home)/".codex/skills/pickbrain"
    claude_dir.mkpath
    codex_dir.mkpath
    cp share/"pickbrain/skills/pickbrain/SKILL.md", claude_dir/"SKILL.md"
    cp share/"pickbrain/skills/pickbrain-codex/SKILL.md", codex_dir/"SKILL.md"
  end

  def caveats
    <<~EOS
      Skill definitions have been installed to:
        ~/.claude/skills/pickbrain/SKILL.md
        ~/.codex/skills/pickbrain/SKILL.md

      First run will ingest and embed your sessions (~7s).
      The database is stored at ~/.claude/pickbrain.db
    EOS
  end

  test do
    assert_match "Usage", shell_output("#{bin}/pickbrain 2>&1")
  end
end
